import logging
import math
from itertools import count

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from uncertainties import ufloat, unumpy as unp

from .errors import LUFactError
from .physics import (
    clean_force,
    local_energy,
    pairwise_distance,
    pairwise_self_distance,
    quantum_force,
)
from .plugins import PLUGINS
from .torchext import argmax_random_choice, assign_where, shuffle_tensor
from .utils import energy_offset

__version__ = '0.3.0'
__all__ = ['sample_wf', 'MetropolisSampler', 'LangevinSampler']

log = logging.getLogger(__name__)


def sample_wf(  # noqa: C901
    wf,
    sampler,
    steps,
    writer=None,
    write_figures=False,
    log_dict=None,
    blocks=None,
    *,
    block_size=10,
    equilibrate=True,
):
    r"""Sample a wave function and accumulate expectation values.

    This is a low-level interface, see :func:`~deepqmc.evaluate` for a high-level
    interface. This iterator iteratively draws samples from the sampler, detects
    when equilibrium is reached, and starts calculating and accumulating
    local energies to get an estimate of the energy. Diagnostics is written into
    the Tensorboard writer, and every full block, the step index, the current
    estimate of the energy, and the sampled electron coordinates are yielded.

    Args:
        wf (:class:`~deepqmc.wf.WaveFunction`): wave function model to be sampled
        sampler (iterator): yields batches of electron coordinate samples
        steps (iterator): yields step indexes
        writer (:class:`torch.utils.tensorboard.writer.SummaryWriter`):
            Tensorboard writer
        log_dict (dict-like): step data will be stored in this dictionary if given
        blocks (list): used as storage of blocks. If not given, the iterator
            uses a local storage.
        block_size (int): size of a block (a sequence of samples)
        equilibrate (bool or int): if false, local energies are calculated and
            accumulated from the first sampling step, if true equilibrium is
            detected automatically, if integer argument, specifies number of
            equilibration steps
    """
    blocks = blocks if blocks is not None else []
    calculating_energy = not equilibrate
    buffer = []
    energy = None
    for step, (rs, log_psis, _, info) in zip(steps, sampler):
        if step == 0:
            dist_means = rs.new_zeros(5 * block_size)
            if not equilibrate:
                yield 0, 'eq'
        dist_means[:-1] = dist_means[1:].clone()
        dist_means[-1] = pairwise_self_distance(rs).mean()
        if not calculating_energy:
            if type(equilibrate) is int:
                if step >= equilibrate:
                    calculating_energy = True
            elif dist_means[0] > 0:
                if dist_means[:block_size].std() < dist_means[-block_size:].std():
                    calculating_energy = True
            if calculating_energy:
                yield step, 'eq'
        if calculating_energy:
            Es_loc = local_energy(rs, wf, keep_graph=False)[0]
            buffer.append(Es_loc)
            if log_dict is not None:
                log_dict['coords'] = rs.cpu().numpy()
                log_dict['E_loc'] = Es_loc.cpu().numpy()
                log_dict['log_psis'] = log_psis.cpu().numpy()
            if 'sample_plugin' in PLUGINS:
                PLUGINS['sample_plugin'](wf, rs, log_dict)
            if len(buffer) == block_size:
                buffer = torch.stack(buffer)
                block = unp.uarray(
                    buffer.mean(dim=0).cpu(),
                    buffer.std(dim=0).cpu() / np.sqrt(len(buffer)),
                )
                blocks.append(block)
                buffer = []
            if not buffer:
                blocks_arr = unp.nominal_values(np.stack(blocks, -1))
                err = blocks_arr.mean(-1).std() / np.sqrt(len(blocks_arr))
                energy = ufloat(blocks_arr.mean(), err)
        if writer:
            if calculating_energy:
                writer.add_scalar('E_loc/mean', Es_loc.mean() - energy_offset, step)
                writer.add_scalar('E_loc/var', Es_loc.var(), step)
                writer.add_scalar('E_loc/min', Es_loc.min(), step)
                writer.add_scalar('E_loc/max', Es_loc.max(), step)
                if not buffer:
                    writer.add_scalar(
                        'E/value', energy.nominal_value - energy_offset, step
                    )
                    writer.add_scalar('E/error', energy.std_dev, step)
            if write_figures:
                from matplotlib.figure import Figure

                fig = Figure(dpi=300)
                ax = fig.subplots()
                ax.hist(log_psis.cpu(), bins=100)
                writer.add_figure('log_psi', fig, step)
                fig = Figure(dpi=300)
                ax = fig.subplots()
                ax.hist(info['age'], bins=100)
                writer.add_figure('age', fig, step)
                if calculating_energy:
                    fig = Figure(dpi=300)
                    ax = fig.subplots()
                    ax.hist(Es_loc.cpu(), bins=100)
                    writer.add_figure('E_loc', fig, step)
                    if not buffer:
                        fig = Figure(dpi=300)
                        ax = fig.subplots()
                        ax.hist(blocks_arr.flatten(), bins=100)
                        writer.add_figure('E_block', fig, step)
        if calculating_energy:
            yield step, energy


def samples_from(sampler, steps):
    xs = zip(*(xs for _, xs in zip(steps, sampler)))
    return tuple(
        torch.stack(x, dim=1) if isinstance(x[0], torch.Tensor) else x for x in xs
    )


class Sampler:
    def __init__(self):
        self.state = {}

    def state_dict(self):
        return self.state.copy()

    def load_state_dict(self, state_dict):
        for name, param in self.state.items():
            value = state_dict[name]
            if isinstance(value, torch.Tensor):
                if value.is_floating_point():
                    value = value.to(param.dtype)
                value = value.to(param.device)
            self.state[name] = value


class MetropolisSampler(Sampler):
    r"""Samples electronic wave functions with vanilla Metropolis--Hastings Monte Carlo.

    An instance of this class is an iterator that yields 2-tuples of electron
    coordinates and wave function values with shapes of :math:`(\cdot,N,3)` and
    :math:`(\cdot)`, respectively.

    Args:
        wf (:class:`~deepqmc.wf.WaveFunction`): wave function to sample from
        rs (:class:`torch.Tensor`:math:`(\cdot,N,3)`): initial positions of the
            Markov-chain walkers
        tau (float): :math:`\tau`, proposal step size
        n_first_certain (int): number of initial steps done with 100% acceptance
        target_acceptance (float): initial step size is automatically adjusted
            to achieve this requested acceptance
        n_discard (int): number of steps in the beginning of the sampling that are
            discarded
        n_decorrelate (int): number of extra steps between yielded samples
        max_age (int): maximum age of a walker without a move after which it is
            moved with 100% acceptance
        log_psi_threshold (float): steps into proposals with log wave function values
            below this threshold are always rejected
    """

    def __init__(
        self,
        wf,
        rs,
        writer=None,
        *,
        tau=0.1,
        n_first_certain=3,
        target_acceptance=0.57,
        n_discard=50,
        n_decorrelate=1,
        max_age=None,
        log_psi_threshold=None,
    ):
        super().__init__()
        self.wf = wf
        self.max_age = max_age
        self.n_first_certain = n_first_certain
        self.log_psi_threshold = log_psi_threshold
        self.target_acceptance = target_acceptance
        self.n_discard = n_discard
        self.n_decorrelate = n_decorrelate
        self.state['rs'] = rs.clone()
        self.state['tau'] = tau
        self.restart()
        self.writer = writer
        self._step_writer = 0

    @property
    def rs(self):
        return self.state['rs']

    @property
    def log_psis(self):
        return self.state['log_psis']

    @property
    def sign_psis(self):
        return self.state['sign_psis']

    @property
    def _ages(self):
        return self.state['ages']

    @property
    def tau(self):
        return self.state['tau']

    def proposal(self):
        return self.rs + torch.randn_like(self.rs) * self.tau

    def acceptance_prob(self, rs):
        with torch.no_grad:
            log_psis, sign_psis = self.wf(rs)
        Ps_acc = torch.exp(2 * (log_psis - self.log_psis))
        # Ps_acc might become 0 or inf, however this does not affect
        # the stability of the remaining code
        return Ps_acc, log_psis, sign_psis

    def extra_vars(self):
        return ()

    def extra_writer(self):
        return ()

    def __len__(self):
        return len(self.rs)

    def __repr__(self):
        return (
            f'<{self.__class__.__name__} sample_size={self.rs.shape[0]} '
            'n_electrons={self.rs.shape[1]} tau={self.tau}>'
        )

    @classmethod
    def from_wf(cls, wf, *, sample_size=2_000, **kwargs):
        """Initialize a sampler with random initial walker positions.

        The walker positions are sampled from Gaussians centered on atoms, with
        charge distribution optionally supplied by the wave function ansatz,
        otherwise taken from the nuclear charges.

        Args:
            wf (:class:`~deepqmc.wf.WaveFunction`): wave function to be sampled from
            sample_size (int): number of Markov-chain walkers
            kwargs: all other arguments are passed to the constructor
        """
        rs = rand_from_mol(wf.mol, sample_size, wf.pop_charges())
        return cls(wf, rs, **kwargs)

    def step(self):
        rs = self.proposal()
        Ps_acc, log_psis, sign_psis, *extra_vars = self.acceptance_prob(rs)
        accepted = Ps_acc > torch.rand_like(Ps_acc)
        if self.log_psi_threshold is not None:
            accepted = accepted & (log_psis > self.log_psi_threshold) | (
                (self.log_psis < self.log_psi_threshold) & (log_psis > self.log_psis)
            )
        if self.max_age is not None:
            accepted = accepted | (self._ages >= self.max_age)
        if self.state['step'] < self.n_first_certain:
            accepted = torch.ones_like(accepted)
        self._ages[accepted] = 0
        self._ages[~accepted] += 1
        acceptance = accepted.type(torch.int).sum().item() / self.rs.shape[0]
        info = {
            'acceptance': acceptance,
            'age': self._ages.cpu().numpy(),
            'tau': self.tau,
        }
        assign_where(
            (self.rs, self.log_psis, self.sign_psis, *self.extra_vars()),
            (rs, log_psis, sign_psis, *extra_vars),
            accepted,
        )
        if self.target_acceptance:
            self.state['tau'] /= self.target_acceptance / max(acceptance, 0.05)
        self.state['step'] += 1
        self._step_writer += 1
        if self.writer:
            self.writer.add_scalar(
                'sampling/log_psis/mean', self.log_psis.mean(), self._step_writer
            )
            self.writer.add_scalar(
                'sampling/dists/mean',
                pairwise_self_distance(self.rs).mean(),
                self._step_writer,
            )
            self.writer.add_scalar('sampling/acceptance', acceptance, self._step_writer)
            self.writer.add_scalar('sampling/tau', self.tau, self._step_writer)
            self.writer.add_scalar(
                'sampling/age/max', info['age'].max(), self._step_writer
            )
            self.writer.add_scalar(
                'sampling/age/rms',
                np.sqrt((info['age'] ** 2).mean()),
                self._step_writer,
            )
            self.extra_writer()
        return self.rs.clone(), self.log_psis.clone(), self.sign_psis.clone(), info

    def iter_with_info(self):
        for i in count(-self.n_discard):
            sample = self.step()
            if i >= 0 and i % (self.n_decorrelate + 1) == 0:
                yield sample

    def __iter__(self):
        for *sample, _ in self.iter_with_info():
            yield sample

    def iter_batches(self, *, epoch_size, batch_size, range=range):
        """Iterate over buffered batches sampled in epochs.

        Each epoch, the wave function is sampled in one shot, the samples
        are buffered, and used to form all batches within a given epoch, entirely
        shuffled.

        Args:
            epoch_size (int): number of batches per epoch
            batch_size (int): number of samples in a batch
            range (callable): alternative to :class:`range`
        """
        n_total = epoch_size * batch_size
        n_steps = math.ceil(n_total / len(self))
        while True:
            xs = samples_from(self, range(n_steps))
            samples_ds = TensorDataset(*(x.flatten(end_dim=1)[:n_total] for x in xs))
            rs_dl = DataLoader(samples_ds, batch_size=batch_size, shuffle=True)
            yield from rs_dl
            self.restart()

    def recompute_psi(self):
        self.state['log_psis'], self.state['sign_psis'] = self.wf(self.rs)

    def restart(self):
        self.state['step'] = 0
        self.recompute_psi()
        self.state['ages'] = torch.zeros_like(self.log_psis, dtype=torch.long)

    def propagate_all(self):
        self.state['rs'] = self.proposal()
        self.restart()


def sort_nucleus_indices(idx, mol):
    # this heuristic takes nuclear indices for placing electrons and sorts them
    # such that the local spin of the electrons is minimized
    dev = idx.device
    selection = idx.bincount(minlength=len(mol.charges))
    available = selection
    n_down = (len(idx) - mol.spin) // 2
    n_nuclei = len(available)
    assigned = torch.tensor([], dtype=torch.long, device=dev)
    # assign core electron pairs to all nuclei with more than one electron
    for j in range(int(available.max()) // 2):
        mask = selection >= 2 * (j + 1)
        if sum(mask).item() <= n_down - len(assigned):
            assigned = torch.cat((assigned, torch.arange(n_nuclei, device=dev)[mask]))
            available -= torch.ones(n_nuclei, device=dev, dtype=torch.long) * 2 * mask
    # order remaining nuclear indices by subsequent closest distances
    dist = pairwise_distance(mol.coords, mol.coords).sort()[1]
    path = (
        torch.tensor([argmax_random_choice(available)], dtype=torch.long, device=dev)
        if sum(available)
        else torch.tensor([], dtype=torch.long, device=dev)
    )
    for _ in range(int(available.sum()) - 1):
        available[path[-1]] -= 1
        path = torch.cat((path, dist[path[-1]][available[dist[path[-1]]] > 0][:1]))
    # assign remaining electrons alternatingly along the path of closest distance
    even, odd = shuffle_tensor(path[0::2]), shuffle_tensor(path[1::2])
    up = shuffle_tensor(torch.cat((assigned, even, odd[n_down - len(assigned) :])))
    down = shuffle_tensor(torch.cat((assigned, odd[: n_down - len(assigned)])))
    return torch.cat((up, down))


def rand_from_mol(mol, bs, pop_charges=None, elec_std=1.0):
    n_atoms = len(mol)
    charges = mol.charges
    n_electrons = (charges.sum() - mol.charge).type(torch.int).item()
    cs = charges
    if pop_charges is not None:
        cs = cs - pop_charges
    base = cs.floor()
    repeats = base.to(torch.long)[None, :].repeat(bs, 1)
    rem = cs - base
    rem_size = int(n_electrons - base.sum())
    if rem_size > 0:
        samples = torch.multinomial(rem.expand(bs, -1), rem_size)
        repeats[
            torch.arange(bs, dtype=torch.long).expand(rem_size, -1).t(), samples
        ] += 1
    idxs = torch.repeat_interleave(
        torch.arange(n_atoms, device=cs.device).expand(bs, -1), repeats.flatten()
    ).view(bs, n_electrons)
    idxs = torch.stack([sort_nucleus_indices(idx, mol) for idx in idxs])
    centers = mol.coords[idxs]
    rs = centers + elec_std * torch.randn_like(centers)
    return rs


class LangevinSampler(MetropolisSampler):
    """Samples electronic wave functions with Langevin Monte Carlo.

    Derived from :class:`MetropolisSampler`.
    """

    @property
    def forces(self):
        return self.state['forces']

    def proposal(self):
        return (
            self.rs
            + self.forces * self.tau
            + torch.randn_like(self.rs) * np.sqrt(self.tau)
        )

    def acceptance_prob(self, rs):
        forces, (log_psis, sign_psis) = self.qforce(rs)
        log_G_ratios = (
            (self.forces + forces)
            * ((self.rs - rs) + self.tau / 2 * (self.forces - forces))
        ).sum(dim=(-1, -2))
        Ps_acc = torch.exp(log_G_ratios + 2 * (log_psis - self.log_psis))
        # Ps_acc might become 0 or inf, however this does not affect
        # the stability of the remaining code
        return Ps_acc, log_psis, sign_psis, forces

    def qforce(self, rs):
        try:
            forces, (log_psis, sign_psis) = quantum_force(rs, self.wf)
        except LUFactError as e:
            e.info['rs'] = rs[e.info['idxs']]
            raise
        # mask out nan forces to increase code stability
        mask = ~log_psis.isneginf()[:, None, None]
        if not mask.all():
            log.warn('Masking forces where psi = 0')
            forces = forces.where(mask, forces.new_tensor(0))
        forces = clean_force(forces, rs, self.wf.mol, tau=self.tau)
        return forces, (log_psis, sign_psis)

    def extra_vars(self):
        return (self.forces,)

    def extra_writer(self):
        self.writer.add_scalar(
            'sampling/forces', self.forces.norm(dim=-1).mean(), self._step_writer
        )

    def recompute_psi(self):
        (
            self.state['forces'],
            (self.state['log_psis'], self.state['sign_psis']),
        ) = self.qforce(self.rs)
