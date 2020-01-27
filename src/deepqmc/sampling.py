import math
from itertools import count

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import torchext
from .physics import clean_force, quantum_force
from .torchext import assign_where, is_cuda

__version__ = '0.2.0'
__all__ = ['MetropolisSampler', 'LangevinSampler']


def samples_from(sampler, steps):
    rs, log_psis, sign_psis, *extra = zip(*(xs for _, xs in zip(steps, sampler)))
    return (
        torch.stack(rs, dim=1),
        torch.stack(log_psis, dim=1),
        torch.stack(sign_psis, dim=1),
        *extra,
    )


class MetropolisSampler:
    r"""Samples electronic wave functions with vanilla Metropolis--Hastings Monte Carlo.

    An instance of this class is an iterator that yields 2-tuples of electron
    coordinates and wave function values with shapes of :math:`(\cdot,N,3)` and
    :math:`(\cdot)`, respectively.

    Args:
        wf (:class:`~deepqmc.wf.WaveFunction`): wave function to sample from
        rs (:class:`torch.Tensor`:math:`(\cdot,N,3)`): initial positions of the
            Markov-chain walkers
        tau (float): :math:`\tau`, proposal step size
        max_age (int): maximum age of a walker without a move after which it is
            moved with 100% acceptance
        n_first_certain (int): number of initial steps done with 100% acceptance
        log_psi_threshold (float): steps into proposals with log wave function values
            below this threshold are always rejected
        target_acceptance (float): initial step size is automatically adjusted
            to achieve this requested acceptance
        n_discard (int): number of steps in the beginning of the sampling that are
            discarded
        n_decorrelate (int): number of extra steps between yielded samples
    """

    def __init__(
        self,
        wf,
        rs,
        *,
        tau=0.1,
        max_age=None,
        n_first_certain=3,
        log_psi_threshold=None,
        target_acceptance=0.57,
        n_discard=50,
        n_decorrelate=1,
        writer=None,
    ):
        self.wf = wf
        self.rs = rs.clone()
        self.tau = tau
        self.max_age = max_age
        self.n_first_certain = n_first_certain
        self.log_psi_threshold = log_psi_threshold
        self.target_acceptance = target_acceptance
        self.n_discard = n_discard
        self.n_decorrelate = n_decorrelate
        self.restart()
        self.writer = writer
        self._totalstep = 0

    def proposal(self):
        return self.rs + torch.randn_like(self.rs) * self.tau

    def acceptance_prob(self, rs):
        with torch.no_grad:
            log_psis, sign_psis = self.wf(rs)
        Ps_acc = torch.exp(2 * (log_psis - self.log_psis))
        # Ps_acc might become 0 or inf, however this does not affect the stability of the remaining code
        return Ps_acc, log_psis, sign_psis

    def extra_vars(self):
        return ()

    def __len__(self):
        return len(self.rs)

    def __repr__(self):
        return (
            f'<{self.__class__.__name__} sample_size={self.rs.shape[0]} '
            'n_electrons={self.rs.shape[1]} tau={self.tau}>'
        )

    @classmethod
    def from_mf(cls, wf, *, sample_size=2_000, mf=None, **kwargs):
        """Initialize a sampler from a HF calculation.

        The initial walker positions are sampled from Gaussians centered
        on atoms, with charge distribution corresponding to the charge analysis
        of the HF wave function.

        Args:
            wf (:class:`~deepqmc.wf.WaveFunction`): wave function to be sampled from
            sample_size (int): number of Markov-chain walkers
            mf (:class:`pyscf.scf.hf.RHF`): HF calculation used to get Mulliken
                partial charges, taken from ``wf.mf`` if not given
            kwargs: all other arguments are passed to the constructor
        """
        rs = rand_from_mf(mf or wf.mf, sample_size)
        if is_cuda(wf):
            rs = rs.cuda()
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
        if self._step < self.n_first_certain:
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
            self.tau /= self.target_acceptance / max(acceptance, 0.05)
        self._step += 1
        self._totalstep += 1
        if self.writer:
            self.writer.add_scalar('sampling/acceptance', acceptance, self._totalstep)
            self.writer.add_scalar('sampling/tau', self.tau, self._totalstep)
        return self.rs.clone(), self.log_psis.clone(), self.sign_psis.clone(), info

    def iter_with_info(self):
        samples = (self.step() for _ in count())
        return (
            sample
            for i, sample in zip(count(-self.n_discard), samples)
            if i >= 0 and i % (self.n_decorrelate + 1) == 0
        )

    def __iter__(self):
        return (
            (rs, log_psis, sign_psis)
            for rs, log_psis, sign_psis, info in self.iter_with_info()
        )

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
        while True:
            n_steps = math.ceil(epoch_size * batch_size / len(self))
            rs, log_psis, sign_psis = samples_from(self, range(n_steps))
            samples_ds = TensorDataset(
                *(x.flatten(end_dim=1) for x in (rs, log_psis, sign_psis))
            )
            rs_dl = DataLoader(
                samples_ds, batch_size=batch_size, shuffle=True, drop_last=True
            )
            yield from rs_dl
            self.restart()

    def recompute_psi(self):
        self.log_psis, self.sign_psis = self.wf(self.rs)

    def restart(self):
        self._step = 0
        self.recompute_psi()
        self._ages = torch.zeros_like(self.log_psis, dtype=torch.long)

    def propagate_all(self):
        self.rs = self.proposal()
        self.restart()


def rand_from_mf(mf, bs, elec_std=1.0, idxs=None):
    mol = mf.mol
    n_atoms = mol.natm
    charges = mol.atom_charges()
    n_electrons = charges.sum() - mol.charge
    cs = torch.tensor(charges - mf.pop(verbose=0)[1]).float()
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
        torch.arange(n_atoms).expand(bs, -1), repeats.flatten()
    ).view(bs, n_electrons)
    idxs = torch.stack([idxs[i, torch.randperm(idxs.shape[-1])] for i in range(bs)])
    centers = torch.tensor(mol.atom_coords()).float()[idxs]
    rs = centers + elec_std * torch.randn_like(centers)
    return rs


class LangevinSampler(MetropolisSampler):
    """Samples electronic wave functions with Langevin Monte Carlo.

    Derived from :class:`MetropolisSampler`.
    """

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
        # Ps_acc might become 0 or inf, however this does not affect the stability of the remaining code
        return Ps_acc, log_psis, sign_psis, forces

    def qforce(self, rs):
        try:
            forces, (log_psis, sign_psis) = quantum_force(rs, self.wf)
        except torchext.LUFactError as e:
            e.info['rs'] = rs[e.info['idxs']]
            raise
        forces = clean_force(forces, rs, self.wf.mol, tau=self.tau)
        return forces, (log_psis, sign_psis)

    def extra_vars(self):
        return (self.forces,)

    def recompute_psi(self):
        self.forces, (self.log_psis, self.sign_psis) = self.qforce(self.rs)
