import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from .errors import NanLoss
from .physics import clean_force, local_energy
from .torchext import normalize_mean, state_dict_copy, weighted_mean_var
from .utils import NULL_DEBUG

__version__ = '0.1.0'
__all__ = ['fit_wf', 'WaveFunctionLoss', 'LossEnergy']


class WaveFunctionLoss(nn.Module):
    r"""Base class for all wave function loss functions.

    Any such loss must be derived from the local energy and wave function
    values, :math:`L(\{E_\text{loc}[\psi],\psi,w\})`, using also
    importance-sampling weights *w*.

    Shape:
        - Input1, :math:`E_\text{loc}[\psi](\mathbf r)`: :math:`(*)`
        - Input2, :math:`\psi(\mathbf r)`: :math:`(*)`
        - Input3, :math:`w(\mathbf r)`: :math:`(*)`
        - Output, *L*: :math:`()`
    """

    pass


class LossVariance(WaveFunctionLoss):
    def forward(self, Es_loc, psis, ws, E_ref=None, p=1):
        assert psis.grad_fn is None
        E0 = E_ref if E_ref is not None else (ws * Es_loc).mean()
        return (ws * (Es_loc - E0).abs() ** p).mean()


class LossEnergy(WaveFunctionLoss):
    r"""Total energy loss function.

    .. math::
        L:=2\mathbb E\big[(E_\text{loc}-\mathbb E[E_\text{loc}])\ln|\psi|\big]

    Taking a derivative of only the logarithm, the resulting gradient is equivalent,
    thanks to the Hermitian property of the Hamiltonian, to the gradient of the
    plain total energy loss function, :math:`\mathbb E[E_\text{loc}]`.
    """

    def forward(self, Es_loc, psis, ws):
        assert Es_loc.grad_fn is None
        self.weights = ws * (Es_loc - (ws * Es_loc).mean())
        return 2 * (self.weights * torch.log(psis.abs())).mean()


def loss_least_squares(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()


def outlier_mask(x, p, q, dim=None):
    x = x.detach()
    dim = dim if dim is not None else -1
    n = x.shape[dim]
    lb = x.kthvalue(int(p * n), dim=dim).values
    ub = x.kthvalue(int((1 - p) * n), dim=dim).values
    return (
        (x - (lb + ub).unsqueeze(dim) / 2).abs() > q * (ub - lb).unsqueeze(dim),
        (lb, ub),
    )


def log_clipped_outliers(x, q):
    x = x.detach()
    median = x.median()
    x = x - median
    a = q * x.abs().mean()
    x = torch.where(
        x.abs() <= a, x, x.sign() * a * (1 + torch.log((1 + (x.abs() / a) ** 2) / 2))
    )
    return median + x


def fit_wf(
    wf,
    loss_func,
    opt,
    sampler,
    steps,
    *,
    require_energy_gradient=False,
    require_psi_gradient=True,
    clip_grad=None,
    writer=None,
    debug=NULL_DEBUG,
    skip_outliers=False,
    clip_outliers=True,
    p=0.01,
    q=5,
    subbatch_size=10_000,
    clean_tau=None,
):
    r"""Fit a wave function using the variational principle and gradient descent.

    This is a low-level interface, see :func:`~deepqmc.train` for a high-level
    interface. This iterator iteratively draws samples from the sampler, evaluates
    the local energy, processes outliers, calculates the loss function, and updates
    the wave function model parameters using a gradient of the loss and an
    optimizer. Diagnostics is written into the Tensorboard writer, and finally
    at the end of each iteration the step index is yielded, so that the caller
    can do some additional processing such as learning rate scheduling.

    Args:
        wf (:class:`~deepqmc.wf.WaveFunction`): wave function model to be fitted
        loss_func (:class:`WaveFunctionLoss`): loss function that accepts local
            energy and wave function values
        opt (:class:`torch.optim.Optimizer`): optimizer
        sampler (iterator): yields batches of electron coordinate samples
        steps (iterator): yields step indexes
        require_energy_gradient (bool): whether the loss function requires
            gradients of the local energy
        require_psi_gradient (bool): whether the loss function requires
            gradients of the wave function
        clip_grad (bool): whether to clip gradients using
            :func:`torch.nn.utils.clip_grad_norm_`
        writer (:class:`torch.utils.tensorboard.writer.SummaryWriter`):
            Tensorboard writer
        skip_outliers (bool): whether to skip local energy outliers
        clip_outliers (bool): whether to clip local energy outliers
        p (float): percentile defining outliers
        q (float): multiple of MAE defining outliers
        subbatch_size (int): number of samples for a single vectorized loss evaluation
        clean_tau (float): if not :data:`None`, :math:`\tau` used for force
            cleaning
    """
    assert not (skip_outliers and clip_outliers)
    for step, (rs, psi0s) in zip(steps, sampler):
        d = debug[step]
        d['psi0s'], d['rs'], d['state_dict'] = psi0s, rs, state_dict_copy(wf)
        subbatch_size = subbatch_size or len(rs)
        subbatches = []
        for rs, psi0s in DataLoader(TensorDataset(rs, psi0s), batch_size=subbatch_size):
            Es_loc, psis, forces = local_energy(
                rs,
                wf,
                create_graph=require_energy_gradient,
                keep_graph=require_psi_gradient,
                return_grad=True,
            )
            ws = (psis.detach() / psi0s) ** 2
            if clean_tau is not None:
                forces = forces / psis.detach()[:, None, None]
                forces_clean = clean_force(forces, rs, wf.mol, tau=clean_tau)
                forces, forces_clean = (
                    x.flatten(start_dim=-2).norm(dim=-1) for x in (forces, forces_clean)
                )
                force_ws = forces_clean / forces
            else:
                force_ws = torch.ones_like(ws)
            total_ws = ws * force_ws
            if skip_outliers:
                outliers = outlier_mask(Es_loc, p, q)[0]
                Es_loc_loss, psis, total_ws = (
                    Es_loc[~outliers],
                    psis[~outliers],
                    total_ws[~outliers],
                )
            elif clip_outliers:
                Es_loc_loss = log_clipped_outliers(Es_loc, q)
            else:
                Es_loc_loss = Es_loc
            loss = loss_func(Es_loc_loss, psis, normalize_mean(total_ws))
            loss.backward()
            subbatches.append(
                (
                    loss.detach().view(1),
                    Es_loc.detach(),
                    Es_loc_loss.detach(),
                    psis.detach(),
                    ws,
                    force_ws,
                    total_ws,
                )
            )
        loss, Es_loc, Es_loc_loss, psis, ws, forces_ws, total_ws = (
            torch.cat(xs) for xs in zip(*subbatches)
        )
        loss = d['loss'] = loss.sum()
        if torch.isnan(loss).any():
            raise NanLoss()
        d['Es_loc'], d['psis'] = Es_loc, psis
        if clip_grad:
            clip_grad_norm_(wf.parameters(), clip_grad)
        if writer:
            E_loc_mean, E_loc_var = weighted_mean_var(Es_loc, ws)
            writer.add_scalar('E_loc/mean', E_loc_mean, step)
            writer.add_scalar('E_loc/var', E_loc_var, step)
            E_loc_loss_mean, E_loc_loss_var = weighted_mean_var(Es_loc_loss, total_ws)
            writer.add_scalar('E_loc_loss/mean', E_loc_loss_mean, step)
            writer.add_scalar('E_loc_loss/var', E_loc_loss_var, step)
            writer.add_scalar('psi/sq_mean', (psis ** 2).mean(), step)
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('weights/mean', ws.mean(), step)
            writer.add_scalar('weights/median', ws.median(), step)
            writer.add_scalar('weights/var', ws.var(), step)
            writer.add_scalar('force_weights/min', force_ws.min(), step)
            writer.add_scalar('force_weights/max', force_ws.max(), step)
            grads = torch.cat(
                [p.grad.flatten() for p in wf.parameters() if p.grad is not None]
            )
            writer.add_scalar('grad/norm', grads.norm(), step)
            for label, value in wf.tracked_parameters():
                writer.add_scalar(f'param/{label}', value, step)
            lr = opt.state_dict()['param_groups'][0]['lr']
            writer.add_scalar('misc/learning_rate', lr, step)
            writer.add_scalar('misc/batch_size', len(Es_loc), step)
        opt.step()
        opt.zero_grad()
        yield step


def fit_wf_supervised(
    fit_net,
    true_net,
    loss_func,
    opt,
    sample_gen,
    correlated_sampling=True,
    acc_grad=1,
    writer=None,
    start=0,
    debug=NULL_DEBUG,
    scheduler=None,
    epoch_size=100,
):
    for step, (rs, psi0s) in enumerate(sample_gen, start=start):
        d = debug[step]
        d['psi0s'], d['rs'] = psi0s, rs
        psis_fit = fit_net(rs)
        psis_true = true_net(rs)
        loss = loss_func(psis_fit, psis_true)
        if writer:
            writer.add_scalar('loss', loss, step)
            for label, value in fit_net.tracked_parameters():
                writer.add_scalar(f'param/{label}', value, step)
        loss.backward()
        if (step + 1) % acc_grad == 0:
            opt.step()
            opt.zero_grad()
        d['state_dict'] = state_dict_copy(fit_net)
        if scheduler and (step + 1) % epoch_size == 0:
            scheduler.step()
