import jax
import jax.numpy as jnp
from jax import lax

from .utils import vec_where

__all__ = ()


class MetropolisSampler:
    def __init__(self, hamil, tau, edge_builder=None):
        self.hamil = hamil
        self.tau = tau
        self.edge_builder = edge_builder

    def _update(self, state, wf):
        edges = self.edge_builder(state['r']) if self.edge_builder else None
        psi = wf(state['r'], edges) if self.edge_builder else wf(state['r'])
        state = {**state, 'psi': psi, **({'edges': edges} if self.edge_builder else {})}
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
        state = jax.tree_map(lambda xp, x: vec_where(accepted, xp, x), prop, state)
        return (state['r'], *((state['edges'],) if self.edge_builder else ())), state


class DecorrSampler:
    def __init__(self, sampler, decorr):
        self.sampler = sampler
        self.decorr = decorr

    def init(self, *args):
        return self.sampler.init(*args)

    def sample(self, state, rng, wf):
        if self.sampler.edge_builder:
            for rng_sample in jax.random.split(rng, self.decorr):
                _, state = self.sampler.sample(state, rng_sample, wf)
        else:
            state, _ = lax.scan(
                lambda state, rng: (
                    self.sampler.sample(state, rng, wf)[1],
                    None,
                ),
                state,
                jax.random.split(rng, self.decorr),
            )
        return (
            state['r'],
            *((state['edges'],) if self.sampler.edge_builder else ()),
        ), state
