from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import lax

from deepqmc.sampling.sampling_utils import clean_force

from ..hamil import MolecularHamiltonian
from ..physics import pairwise_self_distance
from ..types import (
    KeyArray,
    ParametrizedWaveFunction,
    Params,
    PhysicalConfiguration,
    SamplerState,
    Stats,
)
from ..utils import multinomial_resampling, split_dict
from .electron_sample_initializers import ElectronSampleInitializer

__all__ = [
    'MetropolisSampler',
    'LangevinSampler',
    'DecorrSampler',
    'ResampledSampler',
]


class MetropolisSampler:
    r"""
    Metropolis--Hastings Monte Carlo sampler.

    The :meth:`sample` method of this class returns electron coordinate samples
    from the distribution defined by the square of the sampled wave function.

    Args:
        hamil (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the physical
            system.
        wf: the :data:`apply` method of the :data:`haiku` transformed ansatz object.
        tau (float): optional, the proposal step size scaling factor. Adjusted during
            every step if :data:`target_acceptance` is specified.
        target_acceptance (float): optional, if specified the proposal step size
            will be scaled such that the ratio of accepted proposal steps approaches
            :data:`target_acceptance`.
        max_age (int): optional, if specified the next proposed step will always be
            accepted for a walker that hasn't moved in the last :data:`max_age` steps.
    """

    WALKER_STATE = ['r', 'psi', 'age']

    def __init__(
        self,
        hamil: MolecularHamiltonian,
        wf: ParametrizedWaveFunction,
        *,
        sample_initializer: ElectronSampleInitializer,
        tau: float = 1.0,
        target_acceptance: float = 0.57,
        max_age: Optional[int] = None,
    ):
        self.hamil = hamil
        self.sample_initializer = jax.vmap(
            sample_initializer, (0, None, None, None, None, None)
        )
        self.initial_tau = tau
        self.target_acceptance = target_acceptance
        self.max_age = max_age
        self.wf = wf

    def _update(
        self, state: SamplerState, params: Params, R: jax.Array
    ) -> SamplerState:
        psi = jax.vmap(self.wf, (None, 0))(params, self.phys_conf(R, state['r']))
        state = {**state, 'psi': psi}
        return state

    def update(self, state: SamplerState, params: Params, R: jax.Array) -> SamplerState:
        return self._update(state, params, R)

    def init(self, rng: KeyArray, params: Params, n: int, R: jax.Array) -> SamplerState:
        state = {
            'r': self.sample_initializer(
                jax.random.split(rng, n),
                self.hamil.mol.charges,
                self.hamil.ns_valence,
                R,
                self.hamil.n_up,
                self.hamil.n_down,
            ),
            'age': jnp.zeros(n, jnp.int32),
            'tau': jnp.array(self.initial_tau),
        }

        return self._update(state, params, R)

    def _proposal(self, rng: KeyArray, state: SamplerState) -> jax.Array:
        r = state['r']
        return r + state['tau'] * jax.random.normal(rng, r.shape)

    def _acc_log_prob(self, state: SamplerState, prop: SamplerState) -> jax.Array:
        return 2 * (prop['psi'].log - state['psi'].log)

    def _accept(
        self,
        rng: KeyArray,
        state: SamplerState,
        prop: SamplerState,
        log_prob: jax.Array,
        max_age: Optional[int] = None,
        target_acceptance: Optional[float] = None,
    ) -> tuple[SamplerState, jax.Array]:
        accepted = log_prob > jnp.log(jax.random.uniform(rng, log_prob.shape))
        if max_age is not None:
            accepted |= state['age'] >= max_age
        acceptance = accepted.astype(int).sum() / accepted.shape[0]
        prop['tau'] = state['tau'] / (
            target_acceptance / jnp.max(jnp.stack([acceptance, jnp.array(0.05)]))
            if target_acceptance is not None
            else 1
        )
        state['age'] += 1
        prop['age'] = jnp.zeros_like(state['age'])
        (prop, other), (state, _) = (
            split_dict(d, lambda k: k in self.WALKER_STATE) for d in (prop, state)
        )
        state = {
            **jax.tree_util.tree_map(
                lambda xp, x: jax.vmap(jnp.where)(accepted, xp, x), prop, state
            ),
            **other,
        }
        return state, acceptance

    def sample(
        self, rng: KeyArray, state: SamplerState, params: Params, R: jax.Array
    ) -> tuple[SamplerState, PhysicalConfiguration, Stats]:
        rng_prop, rng_acc = jax.random.split(rng)
        prop = self._update(
            {'tau': state['tau'], 'r': self._proposal(rng_prop, state)}, params, R
        )  # type: ignore
        log_prob = self._acc_log_prob(state, prop)
        state, acceptance = self._accept(
            rng_acc, state, prop, log_prob, self.max_age, self.target_acceptance
        )
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

    def phys_conf(self, R: jax.Array, r: jax.Array, **kwargs) -> PhysicalConfiguration:
        if r.ndim == 2:
            return PhysicalConfiguration(R, r, jnp.array(0))  # type: ignore
        n_smpl = len(r)
        return PhysicalConfiguration(
            jnp.tile(R[None], (n_smpl, 1, 1)),  # type: ignore
            r,
            jnp.zeros(n_smpl, dtype=jnp.int32),
        )


class LangevinSampler(MetropolisSampler):
    r"""
    Metropolis adjusted Langevin Monte Carlo sampler.

    Derived from :class:`MetropolisSampler`.

    Args:
        hamil (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the physical
            system.
        wf: the :data:`apply` method of the :data:`haiku` transformed ansatz object.
        tau (float): optional, the proposal step size scaling factor. Adjusted during
            every step if :data:`target_acceptance` is specified.
        target_acceptance (float): optional, if specified the proposal step size
            will be scaled such that the ratio of accepted proposal steps approaches
            :data:`target_acceptance`.
        max_age (int): optional, if specified the next proposed step will always be
            accepted for a walker that hasn't moved in the last :data:`max_age` steps.
    """

    WALKER_STATE = MetropolisSampler.WALKER_STATE + ['force']

    def _update(
        self, state: SamplerState, params: Params, R: jax.Array
    ) -> SamplerState:
        @jax.vmap
        @partial(jax.value_and_grad, has_aux=True)
        def wf_and_force(r):
            psi = self.wf(params, self.phys_conf(R, r))
            return psi.log, psi

        (_, psi), force = wf_and_force(state['r'])
        # Warning: here tau is coming from the previous iteration
        force = clean_force(
            force, self.phys_conf(R, state['r']), self.hamil.mol, tau=state['tau']
        )
        state = {**state, 'psi': psi, 'force': force}
        return state

    def _proposal(
        self,
        rng: KeyArray,
        state: SamplerState,
    ) -> jax.Array:
        r, tau = state['r'], state['tau']
        r = r + tau * state['force'] + jnp.sqrt(tau) * jax.random.normal(rng, r.shape)
        return r

    def _acc_log_prob(self, state: SamplerState, prop: SamplerState) -> jax.Array:
        log_G_ratios = jnp.sum(
            (state['force'] + prop['force'])
            * (
                (state['r'] - prop['r'])
                + state['tau'] / 2 * (state['force'] - prop['force'])
            ),
            axis=tuple(range(1, len(state['r'].shape))),
        )
        return log_G_ratios + 2 * (prop['psi'].log - state['psi'].log)


class OppositeSpinExchangeSampler:
    def __init__(
        self,
        *,
        up_logits_fn: Optional[Callable] = None,
        down_logits_fn: Optional[Callable] = None,
    ):
        self.up_logits_fn = up_logits_fn or self.default_logits_fn
        self.down_logits_fn = down_logits_fn or self.default_logits_fn

    def default_logits_fn(self, r):
        return jnp.zeros(len(r))

    def exchange_proposal(self, rng: KeyArray, state: SamplerState) -> jax.Array:
        rng_up, rng_down = jax.random.split(rng)
        r = state['r']
        batch_idx = jnp.arange(len(r))
        r_up = r[:, : self.hamil.n_up]  # type: ignore
        r_down = r[:, self.hamil.n_up :]  # type: ignore
        up_idx = jax.random.categorical(rng_up, jax.vmap(self.up_logits_fn)(r_up))
        down_idx = jax.random.categorical(
            rng_down, jax.vmap(self.down_logits_fn)(r_down)
        )
        exchanged_up = r_up.at[batch_idx, up_idx].set(r_down[batch_idx, down_idx])
        exchanged_down = r_down.at[batch_idx, down_idx].set(r_up[batch_idx, up_idx])
        return jnp.concatenate([exchanged_up, exchanged_down], axis=1)

    def exchange_acc_log_prob(
        self, state: SamplerState, prop: SamplerState
    ) -> jax.Array:
        return 2 * (prop['psi'].log - state['psi'].log)

    def sample(
        self, rng: KeyArray, state: SamplerState, params: Params, R: jax.Array
    ) -> tuple[SamplerState, PhysicalConfiguration, Stats]:
        rng_super, rng_prop, rng_acc = jax.random.split(rng, 3)
        state, _, stats = super().sample(rng_super, state, params, R)  # type: ignore
        prop = self._update({'r': self.exchange_proposal(rng_prop, state)}, params, R)  # type: ignore
        log_prob = self.exchange_acc_log_prob(state, prop)
        state, exchange_acceptance = self._accept(rng_acc, state, prop, log_prob)  # type: ignore
        stats = {'sampling/exchange_acceptance': exchange_acceptance, **stats}
        return state, self.phys_conf(R, state['r']), stats  # type: ignore


class DecorrSampler:
    r"""
    Insert decorrelating steps into chained samplers.

    This sampler cannot be used as the last element of a sampler chain.

    Args:
        length (int): the samples will be taken in every :data:`length` MCMC step,
            that is, :data:`length` :math:`-1` decorrelating steps are inserted.
    """

    def __init__(self, *, length):
        self.length = length

    def sample(
        self, rng: KeyArray, state: SamplerState, params: Params, R: jax.Array
    ) -> tuple[SamplerState, PhysicalConfiguration, Stats]:
        sample = super().sample  # type: ignore
        state, stats = lax.scan(
            lambda state, rng: sample(rng, state, params, R)[::2],
            state,
            jax.random.split(rng, self.length),
        )
        stats = {k: v[-1] for k, v in stats.items()}
        return state, self.phys_conf(R, state['r']), stats  # type: ignore


class ResampledSampler:
    r"""
    Add resampling to chained samplers.

    This sampler cannot be used as the last element of a sampler chain.
    The resampling is performed by accumulating weights on each MCMC walker
    in each step. Based on a fixed resampling period :data:`period` and/or a
    threshold :data:`threshold` on the normalized effective sample size the walker
    positions are sampled according to the multinomial distribution defined by
    these weights, and the weights are reset to one. Either :data:`period` or
    :data:`threshold` have to be specified.


    Args:
        period (int): optional, if specified the walkers are resampled every
            :data:`period` MCMC steps.
        threshold (float): optional, if specified the walkers are resampled if
            the effective sample size normalized with the batch size is below
            :data:`threshold`.
    """

    def __init__(
        self, *, period: Optional[int] = None, threshold: Optional[float] = None
    ):
        assert period is not None or threshold is not None
        self.period = period
        self.threshold = threshold

    def update(self, state: SamplerState, params: Params, R: jax.Array) -> SamplerState:
        state['log_weight'] -= 2 * state['psi'].log
        state = self._update(state, params, R)  # type: ignore
        state['log_weight'] += 2 * state['psi'].log
        state['log_weight'] -= state['log_weight'].max()
        return state

    def init(self, *args, **kwargs):
        state = super().init(*args, **kwargs)  # type: ignore
        state = {
            **state,
            'step': jnp.array(0),
            'log_weight': jnp.zeros_like(state['psi'].log),
        }
        return state

    def resample_walkers(self, rng_re: KeyArray, state: SamplerState) -> SamplerState:
        idx = multinomial_resampling(rng_re, jnp.exp(state['log_weight']))
        state, other = split_dict(state, lambda k: k in self.WALKER_STATE)  # type: ignore
        state = {
            **jax.tree_util.tree_map(lambda x: x[idx], state),
            **other,
            'step': jnp.array(0),
            'log_weight': jnp.zeros_like(other['log_weight']),
        }
        return state

    def sample(
        self, rng: KeyArray, state: SamplerState, params: Params, R: jax.Array
    ) -> tuple[SamplerState, PhysicalConfiguration, Stats]:
        rng_re, rng_smpl = jax.random.split(rng)
        state, _, stats = super().sample(rng_smpl, state, params, R)  # type: ignore
        state['step'] += 1
        weight = jnp.exp(state['log_weight'])
        ess = jnp.sum(weight) ** 2 / jnp.sum(weight**2)
        stats['sampling/effective sample size'] = ess
        state = jax.lax.cond(
            (self.period is not None and state['step'] >= self.period)
            | (self.threshold is not None and ess / len(weight) < self.threshold),
            self.resample_walkers,
            lambda rng, state: state,
            rng_re,
            state,
        )
        return state, self.phys_conf(R, state['r']), stats  # type: ignore
