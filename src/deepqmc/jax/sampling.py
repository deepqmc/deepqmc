from functools import partial

import jax
import jax.numpy as jnp

__all__ = ()


class MetropolisSampler:
    def __init__(self, hamil, target_acceptance=0.57):
        self.hamil = hamil
        self.target_acceptance = target_acceptance

    def _update(self, state, wf, wf_state):
        psi, wf_state = wf(wf_state, state['r'])
        state = {**state, 'psi': psi, 'wf_state': wf_state}
        return state

    def init(self, rng, wf, wf_state, n, tau=0.1):
        state = {
            'tau': jnp.array(tau),
            'r': self.hamil.init_sample(rng, n),
            'age': jnp.zeros(n, jnp.int32),
        }
        state = self._update(state, wf, wf_state)
        return state

    def _proposal(self, state, rng):
        r = state['r']
        return r + state['tau'] * jax.random.normal(rng, r.shape)

    def _acc_prob(self, state, prop):
        return jnp.exp(2 * (prop['psi'].log - state['psi'].log))

    def sample(self, state, rng, wf):
        rng_prop, rng_acc = jax.random.split(rng)
        prop = {
            'r': self._proposal(state, rng_prop),
            'age': jnp.zeros_like(state['age']),
        }
        prop = self._update(prop, wf, state['wf_state'])
        prob = self._acc_prob(state, prop)
        accepted = prob > jax.random.uniform(rng_acc, prob.shape)
        acceptance = accepted.astype(int).sum() / accepted.shape[0]
        if self.target_acceptance:
            state['tau'] /= self.target_acceptance / jnp.max(
                jnp.stack([acceptance, jnp.array(0.05)])
            )
        state = {**state, 'age': state['age'] + 1}
        tau = state.pop('tau')
        state = jax.tree_util.tree_map(
            lambda xp, x: jax.vmap(jnp.where)(accepted, xp, x), prop, state
        )
        state['tau'] = tau
        stats = {
            'sampling/acceptance': acceptance,
            'sampling/tau': state['tau'],
            'sampling/age/mean': jnp.mean(state['age']),
            'sampling/age/max': jnp.max(state['age']),
            'sampling/log_psi/mean': jnp.mean(state['psi'].log),
            'sampling/log_psi/std': jnp.std(state['psi'].log),
        }
        return state['r'], state, stats


class DecorrSampler:
    def __init__(self, sampler, decorr):
        self.sampler = sampler
        self.decorr = decorr

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.sampler, name)

    def init(self, *args):
        return self.sampler.init(*args)

    def sample(self, state, rng, wf):
        state, _ = jax.lax.scan(
            lambda state, rng: (self.sampler.sample(state, rng, wf)[1], None),
            state,
            jax.random.split(rng, self.decorr),
        )

        return self.sampler.sample(state, rng, wf)


def init_sampling(
    rng,
    hamil,
    ansatz,
    state_callback,
    *,
    sampler=MetropolisSampler,
    sampler_kwargs=None,
    sample_size=2000,
    decorr=1,
):
    sampler = sampler(hamil, **(sampler_kwargs or {}))
    if decorr > 1:
        sampler = DecorrSampler(sampler, decorr=decorr)

    rng_hamil, rng_ansatz, rng_smpl = jax.random.split(rng, 3)
    init_smpl = hamil.init_sample(rng_hamil, sample_size)
    params, wf_state = jax.vmap(ansatz.init, (None, 0), (None, 0))(
        rng_ansatz, init_smpl
    )
    wf = jax.vmap(ansatz.apply, (None, 0, 0))
    init_wf = partial(wf, params)

    smpl_state = sampler.init(rng_smpl, init_wf, wf_state, sample_size)
    if state_callback:
        wf_state, overflow = state_callback(smpl_state['wf_state'])
        if overflow:
            smpl_state = sampler.init(rng_smpl, init_wf, wf_state, sample_size)

    @jax.jit
    def sample_wf(rng, params, smpl_state):
        return sampler.sample(smpl_state, rng, partial(wf, params))

    return params, smpl_state, sample_wf
