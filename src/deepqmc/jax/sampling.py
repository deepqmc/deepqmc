import jax
import jax.numpy as jnp
from jax import lax

__all__ = ()


class MetropolisSampler:
    def __init__(self, hamil, tau):
        self.hamil = hamil
        self.tau = tau

    def _update(self, state, wf):
        state = {**state, 'psi': wf(state['r'])}
        return state

    def init(self, rng, wf, n):
        state = {'r': self.hamil.init_sample(rng, n), 'age': jnp.zeros(n, jnp.int32)}
        state = self._update(state, wf)
        return state

    def sample(self, state, rng, wf):
        rng_prop, rng_acc = jax.random.split(rng)
        r = state['r']
        prop = {
            'r': r + self.tau * jax.random.normal(rng_prop, r.shape),
            'age': jnp.zeros_like(state['age']),
        }
        prop = self._update(prop, wf)
        prob = jnp.exp(2 * (prop['psi'].log - state['psi'].log))
        accepted = prob > jax.random.uniform(rng_acc, prob.shape)
        state = {**state, 'age': state['age'] + 1}
        state = jax.tree_map(
            lambda xp, x: jax.vmap(jnp.where)(accepted, xp, x), prop, state
        )
        return state['r'], state


class DecorrSampler:
    def __init__(self, sampler, decorr):
        self.sampler = sampler
        self.decorr = decorr

    def init(self, *args):
        return self.sampler.init(*args)

    def sample(self, state, rng, wf):
        state, _ = lax.scan(
            lambda state, rng: (self.sampler.sample(state, rng, wf)[1], None),
            state,
            jax.random.split(rng, self.decorr),
        )
        return state['r'], state
