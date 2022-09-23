from functools import partial

import jax
import jax.numpy as jnp

from .physics import pairwise_diffs, pairwise_self_distance
from .utils import multinomial_resampling

__all__ = ()


class MetropolisSampler:
    def __init__(
        self,
        hamil,
        sample_size=2000,
        target_acceptance=0.57,
        decorr=20,
        n_first_certain=0,
        resampling_frequency=0,
        max_age=None,
        tau=0.1,
    ):
        assert not n_first_certain or n_first_certain < decorr
        self.hamil = hamil
        self.sample_size = sample_size
        self.initial_tau = tau
        self.target_acceptance = target_acceptance
        self.decorr = decorr
        self.n_first_certain = n_first_certain
        self.resampling_frequency = resampling_frequency
        self.max_age = max_age

    def _update(self, state, wf, wf_state, log_weights=None):
        psi, wf_state = jax.vmap(wf)(wf_state, state['r'])
        if log_weights is not None:
            state['log_weights'] = log_weights + 2 * (psi.log - state['psi'].log)
        state = {**state, 'psi': psi, 'wf_state': wf_state}
        return state

    def init(self, rng, wf, wf_state):
        state = {
            'step': jnp.array(0),
            'tau': jnp.array(self.initial_tau),
            'r': self.hamil.init_sample(rng, self.sample_size),
            'age': jnp.zeros(self.sample_size, jnp.int32),
            'log_weights': jnp.zeros(self.sample_size),
        }
        state = self._update(state, wf, wf_state)
        return state

    def _proposal(self, state, rng):
        r = state['r']
        return r + state['tau'] * jax.random.normal(rng, r.shape)

    def _acc_log_prob(self, state, prop):
        return 2 * (prop['psi'].log - state['psi'].log)

    def _step(self, state, rng, wf, accept_all=False):
        rng_prop, rng_acc = jax.random.split(rng)
        prop = {
            'r': self._proposal(state, rng_prop),
            'age': jnp.zeros_like(state['age']),
            'tau': state['tau'],
        }
        prop = self._update(prop, wf, state['wf_state'])
        del prop['tau']
        log_prob = self._acc_log_prob(state, prop)
        accepted = jnp.logical_or(
            jnp.broadcast_to(jnp.array([accept_all]), log_prob.shape),
            log_prob > jnp.log(jax.random.uniform(rng_acc, log_prob.shape)),
        )
        if self.max_age:
            accepted = jnp.logical_or(accepted, self.max_age <= state['age'])
        acceptance = accepted.astype(int).sum() / accepted.shape[0]
        if self.target_acceptance:
            state['tau'] /= self.target_acceptance / jnp.max(
                jnp.stack([acceptance, jnp.array(0.05)])
            )
        state = {**state, 'age': state['age'] + 1}
        tmp_state = {key: state.pop(key) for key in ['tau', 'step', 'log_weights']}
        state = {
            **jax.tree_util.tree_map(
                lambda xp, x: jax.vmap(jnp.where)(accepted, xp, x), prop, state
            ),
            **tmp_state,
        }
        ess = jnp.sum(jnp.exp(state['log_weights'])) ** 2 / jnp.sum(
            jnp.exp(state['log_weights']) ** 2
        )
        stats = {
            'sampling/acceptance': acceptance,
            'sampling/tau': state['tau'],
            'sampling/age/mean': jnp.mean(state['age']),
            'sampling/age/max': jnp.max(state['age']),
            'sampling/log_psi/mean': jnp.mean(state['psi'].log),
            'sampling/log_psi/std': jnp.std(state['psi'].log),
            'sampling/effective sample size': ess,
            'sampling/dists/mean': jnp.mean(pairwise_self_distance(state['r'])),
        }
        return state['r'], state, stats

    def resample_walkers(self, state, rng):
        weights = jnp.exp(state['log_weights'])
        idxs = multinomial_resampling(rng, weights)
        tmp_state = {key: state.pop(key) for key in ['tau', 'step']}
        state = {
            **jax.tree_util.tree_map(lambda x: x[idxs], state),
            **tmp_state,
            'log_weights': jnp.zeros_like(weights),
            'step': jnp.array(0),
        }
        return state

    def sample(self, state, rng, wf):
        state['step'] = state['step'] + 1
        self._update(
            state,
            wf,
            state['wf_state'],
            state['log_weights'] if self.resampling_frequency else None,
        )
        if self.resampling_frequency:
            state = jax.lax.cond(
                self.resampling_frequency <= state['step'],
                self.resample_walkers,
                lambda state, rng: state,
                state,
                rng,
            )
        if self.n_first_certain:
            rng, rng_certain = jax.random.split(rng)
            state, _ = jax.lax.scan(
                lambda state, rng: (
                    self._step(state, rng, wf, accept_all=True)[1],
                    None,
                ),
                state,
                jax.random.split(rng_certain, self.n_first_certain),
            )
        if self.decorr - self.n_first_certain:
            state, _ = jax.lax.scan(
                lambda state, rng: (self._step(state, rng, wf)[1], None),
                state,
                jax.random.split(rng, self.decorr - self.n_first_certain),
            )
        return self._step(state, rng, wf)


class LangevinSampler(MetropolisSampler):
    def _update(self, state, wf, wf_state, log_weights=None):
        assert log_weights is None  # TODO

        @jax.vmap
        @partial(jax.value_and_grad, argnums=1, has_aux=True)
        def wf_and_force(state, r):
            psi, state = wf(state, r)
            return psi.log, (psi, state)

        (_, (psi, wf_state)), force = wf_and_force(wf_state, state['r'])
        force = clean_force(force, state['r'], self.hamil.mol, tau=state['tau'])
        state = {**state, 'psi': psi, 'wf_state': wf_state, 'force': force}
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


def clean_force(force, r, mol, *, tau):
    z, idx = diffs_to_nearest_nuc(jnp.reshape(r, (-1, 3)), mol.coords)
    a = crossover_parameter(z, jnp.reshape(force, (-1, 3)), mol.charges[idx])
    z, a = jnp.reshape(z, (len(r), -1, 4)), jnp.reshape(a, (len(r), -1))
    av2tau = a * jnp.sum(force**2, axis=-1) * tau
    # av2tau can be small or zero, so the following expression must handle that
    factor = 2 / (jnp.sqrt(1 + 2 * av2tau) + 1)
    force = factor[..., None] * force
    eps = jnp.finfo(r.dtype).eps
    norm_factor = jnp.minimum(
        1.0,
        jnp.sqrt(z[..., -1])
        / (tau * jnp.clip(jnp.linalg.norm(force, axis=-1), eps, None)),
    )
    force = force * norm_factor[..., None]
    return force


def init_sampling(
    rng,
    hamil,
    ansatz,
    state_callback,
    *,
    params=None,
    sampler=MetropolisSampler,
    sampler_kwargs=None,
):
    sampler = sampler(hamil, **(sampler_kwargs or {}))

    rng_hamil, rng_ansatz, rng_smpl = jax.random.split(rng, 3)
    init_smpl = hamil.init_sample(rng_hamil, sampler.sample_size)
    maybe_params, wf_state = jax.vmap(ansatz.init, (None, 0), (None, 0))(
        rng_ansatz, init_smpl
    )
    params = params or maybe_params
    wf = partial(ansatz.apply, params)

    smpl_state = sampler.init(rng_smpl, wf, wf_state)
    if state_callback:
        wf_state, overflow = state_callback(smpl_state['wf_state'])
        if overflow:
            smpl_state = sampler.init(rng_smpl, wf, wf_state)

    @jax.jit
    def sample_wf(rng, params, smpl_state):
        return sampler.sample(smpl_state, rng, partial(ansatz.apply, params))

    return params, smpl_state, sample_wf
