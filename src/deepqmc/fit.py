import logging
from contextlib import nullcontext
from functools import partial

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from uncertainties import ufloat

from .errors import DeepQMCError, NanGradients, NanLoss
from .physics import local_energy
from .torchext import is_cuda, normalize_mean, state_dict_copy, weighted_mean_var
from .utils import NULL_DEBUG, estimate_optimal_batch_size_cuda

__version__ = '0.1.0'
__all__ = ['fit_wf', 'WaveFunctionLoss', 'LossEnergy']

log = logging.getLogger(__name__)


class WaveFunctionLoss(nn.Module):
    r"""Base class for all wave function loss functions.

    Any such loss must be derived from the local energy and wave function
    values, :math:`L(\{E_\text{loc}[\psi],\ln|\psi|,w\})`, using also
    importance-sampling weights *w*.

    Shape:
        - Input1, :math:`E_\text{loc}[\psi](\mathbf r)`: :math:`(*)`
        - Input2, :math:`\ln|\psi(\mathbf r)|`: :math:`(*)`
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

    def forward(self, Es_loc, log_psis, ws):
        assert Es_loc.grad_fn is None
        self.weights = 2 * (Es_loc - (ws * Es_loc).mean()) / len(Es_loc)
        return (self.weights * ws * log_psis).sum()


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


def fit_wf(  # noqa: C901
    wf,
    loss_func,
    opt,
    sampler,
    steps,
    writer=None,
    log_dict=None,
    debug=NULL_DEBUG,
    require_energy_gradient=False,
    require_psi_gradient=True,
    subbatch_size=None,
    max_memory=None,
    *,
    clip_outliers=True,
    q=5,
    max_grad_norm=None,
    kfac=None,
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
        max_grad_norm (float): maximum gradient norm passed to
            :func:`torch.nn.utils.clip_grad_norm_`
        writer (:class:`torch.utils.tensorboard.writer.SummaryWriter`):
            Tensorboard writer
        log_dict (dict-like): batch data will be stored in this dictionary if given
        clip_outliers (bool): whether to clip local energy outliers
        q (float): multiple of MAE defining outliers
        subbatch_size (int): number of samples for a single vectorized loss evaluation.
            If None and on a GPU, subbatch_size is estimated, else if None and on a CPU,
            no subbatching is done.
        max_memory (float): maximum amount of allocated GPU memory (MiB) to be
            considered if automatically estimating the subbatch_size. If :data:`None`
            and subbatch_size is estimated, the maximum memory is set to the total
            free GPU memory. When training on CPU always set to :data:`None`.
    """
    if not is_cuda(wf) and max_memory:
        raise DeepQMCError(
            'Automatic subbatch_size estimation only implemented for GPU. '
            'When training on CPU, do not use max_memory.'
        )
    elif is_cuda(wf) and not subbatch_size:
        subbatch_size = estimate_optimal_batch_size_cuda(
            partial(fit_wf_mem_test_func, wf, loss_func, require_psi_gradient),
            torch.linspace(200, 500, 4) / (wf.n_up + wf.n_down),
            max_memory=max_memory,
        )
        log.info(f'estimated optimal subbatch size: {subbatch_size}')
    for step, (rs, log_psi0s, sign_psi0s) in zip(steps, sampler):
        opt.zero_grad()
        d = debug[step]
        d['log_psi0s'], d['sign_psi0s'], d['rs'], d['state_dict'] = (
            log_psi0s,
            sign_psi0s,
            rs,
            state_dict_copy(wf),
        )
        subbatch_size = subbatch_size or len(rs)
        subbatches = []
        for rs, log_psi0s, _ in DataLoader(
            TensorDataset(rs, log_psi0s, sign_psi0s), batch_size=subbatch_size
        ):
            with kfac.track_forward() if kfac else nullcontext():
                Es_loc, log_psis, sign_psis = local_energy(
                    rs,
                    wf,
                    create_graph=require_energy_gradient,
                    keep_graph=require_psi_gradient,
                )
            log_ws = 2 * log_psis.detach() - 2 * log_psi0s
            Es_loc_loss = log_clipped_outliers(Es_loc, q) if clip_outliers else Es_loc
            loss = loss_func(Es_loc_loss, log_psis, normalize_mean(log_ws.exp()))
            with kfac.track_backward() if kfac else nullcontext():
                loss.backward()
            if kfac:
                kfac.step_update(loss_func.weights)
            subbatches.append(
                (
                    loss.detach().view(1),
                    Es_loc.detach(),
                    Es_loc_loss.detach(),
                    log_psis.detach(),
                    sign_psis.detach(),
                    log_ws,
                )
            )
        loss, Es_loc, Es_loc_loss, log_psis, sign_psis, log_ws = (
            torch.cat(xs) for xs in zip(*subbatches)
        )
        if torch.isnan(loss).any():
            raise NanLoss()
        if any(
            torch.isnan(p.grad).any() for p in wf.parameters() if p.grad is not None
        ):
            raise NanGradients()
        loss = d['loss'] = loss.sum()
        d['Es_loc'], d['log_psis'], d['sign_psis'] = Es_loc, log_psis, sign_psis
        if max_grad_norm is not None:
            clip_grad_norm_(wf.parameters(), max_grad_norm)
        if writer:
            E_loc_mean, E_loc_var = weighted_mean_var(Es_loc, log_ws.exp())
            E_loc_err = torch.sqrt(E_loc_var / len(Es_loc))
            writer.add_scalar('E_loc/mean', E_loc_mean, step)
            writer.add_scalar('E_loc/var', E_loc_var, step)
            writer.add_scalar('E_loc/min', Es_loc.min(), step)
            writer.add_scalar('E_loc/max', Es_loc.max(), step)
            writer.add_scalar('E_loc/err', E_loc_err, step)
            E_loc_loss_mean, E_loc_loss_var = weighted_mean_var(
                Es_loc_loss, log_ws.exp()
            )
            writer.add_scalar('E_loc_loss/mean', E_loc_loss_mean, step)
            writer.add_scalar('E_loc_loss/var', E_loc_loss_var, step)
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('log_weights/std', log_ws.std(), step)
            grads = torch.cat(
                [p.grad.flatten() for p in wf.parameters() if p.grad is not None]
            )
            writer.add_scalar('grad/norm', grads.norm(), step)
            for label, value in wf.tracked_parameters():
                writer.add_scalar(f'param/{label}', value, step)
            lr = opt.state_dict()['param_groups'][0]['lr']
            writer.add_scalar('misc/learning_rate', lr, step)
            writer.add_scalar('misc/batch_size', len(Es_loc), step)
        if log_dict is not None:
            log_dict['E_loc'] = Es_loc.cpu().numpy()
            log_dict['E_loc_loss'] = Es_loc_loss.cpu().numpy()
            log_dict['log_psis'] = log_psis.cpu().numpy()
            log_dict['sign_psis'] = sign_psis.cpu().numpy()
            log_dict['log_ws'] = log_ws.cpu().numpy()
        if kfac:
            kfac.step_precondition()
            fnorm, gnorm = kfac.step_rescale()
            writer.add_scalar('kfac/KL', fnorm, step)
            writer.add_scalar('kfac/dE', -gnorm, step)
        opt.step()
        yield step, ufloat(E_loc_mean.item(), E_loc_err.item())


def fit_wf_mem_test_func(wf, loss_func, require_psi_gradient, size):
    # require_energy_gradient isn't needed here because it adds only little
    # extra memory to the probe calculation
    assert is_cuda(wf)
    rs = torch.randn((size, wf.n_down + wf.n_up, 3), device='cuda', requires_grad=True)
    E_loc, log_psi, _ = local_energy(rs, wf, keep_graph=require_psi_gradient)
    loss = loss_func(
        E_loc.detach() if require_psi_gradient else E_loc,
        log_psi,
        torch.ones(len(rs)).cuda(),
    )
    loss.backward()


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
