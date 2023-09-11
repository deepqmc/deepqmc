import logging
from functools import partial
from statistics import mean, stdev
from typing import Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jax import lax

from .parallel import replicate_on_devices, rng_iterator
from .physics import pairwise_diffs, pairwise_self_distance
from .types import PhysicalConfiguration
from .utils import multinomial_resampling, split_dict

__all__ = [
    'MetropolisSampler',
    'LangevinSampler',
    'DecorrSampler',
    'ResampledSampler',
    'chain',
]

log = logging.getLogger(__name__)


class Sampler:
    r"""Base class for all QMC samplers."""

    def init(self, rng, wf, n):
        raise NotImplementedError

    def sample(self, rng, state, wf):
        raise NotImplementedError


class MetropolisSampler(Sampler):
    r"""
    Metropolis--Hastings Monte Carlo sampler.

    The :meth:`sample` method of this class returns electron coordinate samples
    from the distribution defined by the square of the sampled wave function.

    Args:
        hamil (jax.hamil.Hamiltonian): the Hamiltonian of the physical system
        tau (float): optional, the proposal step size scaling factor. Adjusted during
            every step if :data:`target_acceptance` is specified.
        target_acceptance (float): optional, if specified the proposal step size
            will be scaled such that the ratio of accepted proposal steps approaches
            :data:`target_acceptance`.
        max_age (int): optional, if specified the next proposed step will always be
            accepted for a walker that hasn't moved in the last :data:`max_age` steps.
    """

    WALKER_STATE = ['r', 'psi', 'age']

    def __init__(self, hamil, *, tau=1.0, target_acceptance=0.57, max_age=None):
        self.hamil = hamil
        self.initial_tau = tau
        self.target_acceptance = target_acceptance
        self.max_age = max_age

    def _update(self, state, wf, R):
        psi = jax.vmap(wf)(self.phys_conf(R, state['r']))
        state = {**state, 'psi': psi}
        return state

    def update(self, state, wf, R):
        return self._update(state, wf, R)

    def init(self, rng, wf, n, R):
        state = {
            'r': self.hamil.init_sample(rng, R, n).r,
            'age': jnp.zeros(n, jnp.int32),
            'tau': jnp.array(self.initial_tau),
        }

        return self._update(state, wf, R)

    def _proposal(self, state, rng):
        r = state['r']
        return r + state['tau'] * jax.random.normal(rng, r.shape)

    def _acc_log_prob(self, state, prop):
        return 2 * (prop['psi'].log - state['psi'].log)

    def sample(self, rng, state, wf, R):
        rng_prop, rng_acc = jax.random.split(rng)
        prop = {
            'r': self._proposal(state, rng_prop),
            'age': jnp.zeros_like(state['age']),
            **{k: v for k, v in state.items() if k not in self.WALKER_STATE},
        }
        prop = self._update(prop, wf, R)
        log_prob = self._acc_log_prob(state, prop)
        accepted = log_prob > jnp.log(jax.random.uniform(rng_acc, log_prob.shape))
        if self.max_age:
            accepted = accepted | (state['age'] >= self.max_age)
        acceptance = accepted.astype(int).sum() / accepted.shape[0]
        if self.target_acceptance:
            prop['tau'] /= self.target_acceptance / jnp.max(
                jnp.stack([acceptance, jnp.array(0.05)])
            )
        state = {**state, 'age': state['age'] + 1}
        (prop, other), (state, _) = (
            split_dict(d, lambda k: k in self.WALKER_STATE) for d in (prop, state)
        )
        state = {
            **jax.tree_util.tree_map(
                lambda xp, x: jax.vmap(jnp.where)(accepted, xp, x), prop, state
            ),
            **other,
        }
        stats = {
            'sampling/acceptance': acceptance,
            'sampling/tau': state['tau'],
            'sampling/age/mean': jnp.mean(state['age']),
            'sampling/age/max': jnp.max(state['age']),
            'sampling/log_psi/mean': jnp.mean(state['psi'].log),
            'sampling/log_psi/std': jnp.std(state['psi'].log),
            'sampling/dists/mean': jnp.mean(pairwise_self_distance(state['r'])),
        }
        return state, self.phys_conf(R, state['r']), stats

    def phys_conf(self, R, r, **kwargs):
        if r.ndim == 2:
            return PhysicalConfiguration(R, r, jnp.array(0))
        n_smpl = len(r)
        return PhysicalConfiguration(
            jnp.tile(R[None], (n_smpl, 1, 1)),
            r,
            jnp.zeros(n_smpl, dtype=jnp.int32),
        )


class LangevinSampler(MetropolisSampler):
    r"""
    Langevin Monte Carlo sampler.

    Derived from :class:`MetropolisSampler`.
    """

    WALKER_STATE = MetropolisSampler.WALKER_STATE + ['force']

    def _update(self, state, wf, R):
        @jax.vmap
        @partial(jax.value_and_grad, has_aux=True)
        def wf_and_force(r):
            psi = wf(self.phys_conf(R, r))
            return psi.log, psi

        (_, psi), force = wf_and_force(state['r'])
        force = clean_force(
            force, self.phys_conf(R, state['r']), self.hamil.mol, tau=state['tau']
        )
        state = {**state, 'psi': psi, 'force': force}
        return state

    def _proposal(self, state, rng):
        r, tau = state['r'], state['tau']
        r = r + tau * state['force'] + jnp.sqrt(tau) * jax.random.normal(rng, r.shape)
        return r

    def _acc_log_prob(self, state, prop):
        log_G_ratios = jnp.sum(
            (state['force'] + prop['force'])
            * (
                (state['r'] - prop['r'])
                + state['tau'] / 2 * (state['force'] - prop['force'])
            ),
            axis=tuple(range(1, len(state['r'].shape))),
        )
        return log_G_ratios + 2 * (prop['psi'].log - state['psi'].log)


class DecorrSampler(Sampler):
    r"""
    Insert decorrelating steps into chained samplers.

    This sampler cannot be used as the last element of a sampler chain.

    Args:
        length (int): the samples will be taken in every :data:`length` MCMC step,
            that is, :data:`length` :math:`-1` decorrelating steps are inserted.
    """

    def __init__(self, *, length):
        self.length = length

    def sample(self, rng, state, wf, R):
        sample = super().sample  # lax cannot parse super()
        state, stats = lax.scan(
            lambda state, rng: sample(rng, state, wf, R)[::2],
            state,
            jax.random.split(rng, self.length),
        )
        stats = {k: v[-1] for k, v in stats.items()}
        return state, self.phys_conf(R, state['r']), stats


class ResampledSampler(Sampler):
    r"""
    Add resampling to chained samplers.

    This sampler cannot be used as the last element of a sampler chain.
    The resampling is performed by accumulating weights on each MCMC walker
    in each step. Based on a fixed resampling period :data:`period` and/or a
    threshold :data:`treshold` on the normalized effective sample size the walker
    positions are sampled according to the multinomial distribution defined by
    these weights, and the weights are reset to one. Either :data:`period` or
    :data:`treshold` have to be specified.


    Args:
        period (int): optional, if specified the walkers are resampled every
            :data:`period` MCMC steps.
        treshold (float): optional, if specified the walkers are resampled if
            the effective sample size normalized with the batch size is below
            :data:`treshold`.
    """

    def __init__(self, *, period=None, treshold=None):
        assert period is not None or treshold is not None
        self.period = period
        self.treshold = treshold

    def update(self, state, wf, R):
        state['log_weight'] -= 2 * state['psi'].log
        state = self._update(state, wf, R)
        state['log_weight'] += 2 * state['psi'].log
        state['log_weight'] -= state['log_weight'].max()
        return state

    def init(self, *args, **kwargs):
        state = super().init(*args, **kwargs)
        state = {
            **state,
            'step': jnp.array(0),
            'log_weight': jnp.zeros_like(state['psi'].log),
        }
        return state

    def resample_walkers(self, rng_re, state):
        idx = multinomial_resampling(rng_re, jnp.exp(state['log_weight']))
        state, other = split_dict(state, lambda k: k in self.WALKER_STATE)
        state = {
            **jax.tree_util.tree_map(lambda x: x[idx], state),
            **other,
            'step': jnp.array(0),
            'log_weight': jnp.zeros_like(other['log_weight']),
        }
        return state

    def sample(self, rng, state, wf, R):
        rng_re, rng_smpl = jax.random.split(rng)
        state, _, stats = super().sample(rng_smpl, state, wf, R)
        state['step'] += 1
        weight = jnp.exp(state['log_weight'])
        ess = jnp.sum(weight) ** 2 / jnp.sum(weight**2)
        stats['sampling/effective sample size'] = ess
        state = jax.lax.cond(
            (self.period is not None and state['step'] >= self.period)
            | (self.treshold is not None and ess / len(weight) < self.treshold),
            self.resample_walkers,
            lambda rng, state: state,
            rng_re,
            state,
        )
        return state, self.phys_conf(R, state['r']), stats


class MultimoleculeSampler(Sampler):
    def __init__(self, sampler, mols, mol_idx_factory=None):
        self.sampler = sampler
        self.mols = mols if isinstance(mols, Sequence) else [mols]

        class MolIdxFactory:
            @staticmethod
            def max_per_mol(sample_size, n_mol):
                assert not (sample_size % n_mol)
                return n_mol * [sample_size // n_mol]

            @staticmethod
            def n_per_mol(sample_size, n_mol, *args, **kwargs):
                assert not (sample_size % n_mol)
                return n_mol * [sample_size // n_mol]

        self.mol_idx_factory = mol_idx_factory or MolIdxFactory()

    def init(self, rng, wf, n):
        wfs = self.assign_wfs(wf)
        sample_sizes = self.mol_idx_factory.max_per_mol(n, len(self))
        states = [
            self.sampler.init(rng, wf, sample_size, mol.coords)
            for rng, wf, sample_size, mol in zip(
                hk.PRNGSequence(rng), wfs, sample_sizes, self.mols
            )
        ]
        return states

    def sample(self, rng, state, wave_function, select_idxs):
        phys_confs, stats = [], []
        wfs = self.assign_wfs(wave_function)
        for i, rng in zip(range(len(self)), hk.PRNGSequence(rng)):
            state[i], phys_conf, stat = self.sampler.sample(
                rng, state[i], wfs[i], self.mols[i].coords
            )
            phys_confs.append(phys_conf)
            stats.append({'per_mol': stat})

        phys_conf = self.join_mols(phys_confs, select_idxs)
        stats = self.join_mols(stats, None)
        return state, phys_conf, stats

    def assign_wfs(self, wf):
        r"""Assign WF models to all molecules.

        This allows for using different WFs for each molecule,
        useful e.g. with HF baselines.
        """
        return wf if isinstance(wf, Sequence) else len(self) * [wf]

    def update(self, states, wf):
        wfs = self.assign_wfs(wf)
        return [
            self.sampler.update(state, wf, mol.coords)
            for state, wf, mol in zip(states, wfs, self.mols)
        ]

    def get_state(self, key, states, select_idxs, default=None):
        try:
            data = [state[key] for state in states]
        except KeyError:
            return default
        data = self.join_mols(data, select_idxs)
        return data

    def join_mols(self, data_per_mol, select_idxs):
        if all(isinstance(d, PhysicalConfiguration) for d in data_per_mol):
            # phys_conf is special because we need to store which molecule the samples
            # are coming from
            data_per_mol = [
                jdc.replace(pc, mol_idx=i * jnp.ones(len(pc), dtype=jnp.int32))
                for i, pc in enumerate(data_per_mol)
            ]
        if select_idxs is None:
            # data is zero dimensional
            return jax.tree_util.tree_map(
                lambda *xs: jnp.concatenate([x[select_idxs] for x in xs]),
                *data_per_mol,
            )
        return jax.tree_util.tree_map(
            lambda *xs: jnp.concatenate(xs)[select_idxs], *data_per_mol
        )

    def phys_conf(self, states, select_idxs):
        phys_confs = [
            self.sampler.phys_conf(mol.coords, state['r'])
            for state, mol in zip(states, self.mols)
        ]
        return self.join_mols(phys_confs, select_idxs)

    def select_idxs(self, sample_size, *args, **kwargs):
        n_smpl_per_mol = self.mol_idx_factory.n_per_mol(
            sample_size, len(self), *args, **kwargs
        )
        max_smpl_per_mol = self.mol_idx_factory.max_per_mol(sample_size, len(self))
        assert all(n <= m for n, m in zip(n_smpl_per_mol, max_smpl_per_mol))
        start_pos = 0
        idxs = []
        for n, m in zip(n_smpl_per_mol, max_smpl_per_mol):
            idxs.append(start_pos + jnp.arange(n))
            start_pos += m
        return jnp.concatenate(idxs)

    def mol_idx(self, sample_size, *args, **kwargs):
        n_smpl_per_mol = self.mol_idx_factory.n_per_mol(
            sample_size, len(self), *args, **kwargs
        )
        assert all(
            n <= m
            for n, m in zip(
                n_smpl_per_mol,
                self.mol_idx_factory.max_per_mol(sample_size, len(self)),
            )
        )
        return jnp.concatenate(
            [
                i * jnp.ones(n_smpl, dtype=jnp.int32)
                for i, n_smpl in enumerate(n_smpl_per_mol)
            ]
        )

    def __len__(self):
        return len(self.mols)


def chain(*samplers):
    r"""
    Combine multiple sampler types, to create advanced sampling schemes.

    For example :data:`chain(DecorrSampler(10),MetropolisSampler(hamil, tau=1.))`
    will create a :class:`MetropolisSampler`, where the samples
    are taken from every 10th MCMC step. The last element of the sampler chain has
    to be either a :class:`MetropolisSampler` or a :class:`LangevinSampler`.

    Args:
        samplers (~jax.sampling.Sampler): one or more sampler instances to combine.

    Returns:
        ~jax.sampling.Sampler: the combined sampler.
    """
    name = 'Sampler'
    bases = tuple(map(type, samplers))
    for base in bases:
        name = name.replace('Sampler', base.__name__)
    chained = type(name, bases, {'__init__': lambda self: None})()
    for sampler in samplers:
        chained.__dict__.update(sampler.__dict__)
    return chained


def diffs_to_nearest_nuc(r, coords):
    z = pairwise_diffs(r, coords)
    idx = jnp.argmin(z[..., -1], axis=-1)
    return z[jnp.arange(len(r)), idx], idx


def crossover_parameter(z, f, charge):
    z, z2 = z[..., :3], z[..., 3]
    eps = jnp.finfo(f.dtype).eps
    z_unit = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
    f_unit = f / jnp.clip(jnp.linalg.norm(f, axis=-1, keepdims=True), eps, None)
    Z2z2 = charge**2 * z2
    return (1 + jnp.sum(f_unit * z_unit, axis=-1)) / 2 + Z2z2 / (10 * (4 + Z2z2))


def clean_force(force, phys_conf, mol, *, tau):
    z, idx = jax.vmap(diffs_to_nearest_nuc)(phys_conf.r, phys_conf.R)
    a = crossover_parameter(z, force, mol.charges[idx])
    av2tau = a * jnp.sum(force**2, axis=-1) * tau
    # av2tau can be small or zero, so the following expression must handle that
    factor = 2 / (jnp.sqrt(1 + 2 * av2tau) + 1)
    force = factor[..., None] * force
    eps = jnp.finfo(phys_conf.r.dtype).eps
    norm_factor = jnp.minimum(
        1.0,
        jnp.sqrt(z[..., -1])
        / (tau * jnp.clip(jnp.linalg.norm(force, axis=-1), eps, None)),
    )
    force = force * norm_factor[..., None]
    return force


def equilibrate(
    rng,
    wf,
    sampler,
    state,
    criterion,
    steps,
    sample_size,
    *,
    block_size,
    n_blocks=5,
):
    criterion = jax.jit(criterion)

    @jax.pmap
    def sample_wf(rng, state, select_idxs):
        return sampler.sample(rng, state, wf, select_idxs)

    buffer_size = block_size * n_blocks
    buffer = []
    for step, rng in zip(steps, rng_iterator(rng)):
        select_idxs = sampler.select_idxs(sample_size // jax.device_count(), step)
        select_idxs = replicate_on_devices(select_idxs)
        state, phys_conf, stats = sample_wf(rng, state, select_idxs)
        yield step, state, stats
        buffer = [*buffer[-buffer_size + 1 :], criterion(phys_conf).item()]
        if len(buffer) < buffer_size:
            continue
        b1, b2 = buffer[:block_size], buffer[-block_size:]
        if abs(mean(b1) - mean(b2)) < min(stdev(b1), stdev(b2)):
            break
