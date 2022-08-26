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

    def sample(self, state, rng, wf):
        rng_prop, rng_acc = jax.random.split(rng)
        r, tau = state['r'], state['tau']
        # tree_map(where, prop, state) would fail with tau in state
        state = {k: v for k, v in state.items() if k != 'tau'}
        prop = {
            'r': r + tau * jax.random.normal(rng_prop, r.shape),
            'age': jnp.zeros_like(state['age']),
        }
        prop = self._update(prop, wf, state['wf_state'])
        prob = jnp.exp(2 * (prop['psi'].log - state['psi'].log))
        accepted = prob > jax.random.uniform(rng_acc, prob.shape)
        if self.target_acceptance:
            acceptance = accepted.astype(int).sum() / accepted.shape[0]
            tau /= self.target_acceptance / jnp.max(
                jnp.stack([acceptance, jnp.array(0.05)])
            )
        state = {**state, 'age': state['age'] + 1}
        state = jax.tree_util.tree_map(
            lambda xp, x: jax.vmap(jnp.where)(accepted, xp, x), prop, state
        )
        return (
            state['r'],
            {
                **state,
                'tau': tau,
            },
        )


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
        return state['r'], state
