import jax
import jax.numpy as jnp
from jax import lax

from .utils import vec_where

__all__ = ()


class MetropolisSampler:
    def __init__(self, hamil, edge_builder=None, target_acceptance=0.57):
        self.hamil = hamil
        self.edge_builder = edge_builder
        self.target_acceptance = target_acceptance

    def _update(self, state, wf):
        edges = self.edge_builder(state['r']) if self.edge_builder else None
        psi = wf(state['r'], edges) if self.edge_builder else wf(state['r'])
        state = {**state, 'psi': psi, **({'edges': edges} if self.edge_builder else {})}
        return state

    def init(self, rng, wf, n, tau=0.1):
        state = {
            'tau': tau,
            'r': self.hamil.init_sample(rng, n),
            'age': jnp.zeros(n, jnp.int32),
        }
        state = self._update(state, wf)
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
        prop = self._update(prop, wf)
        prob = jnp.exp(2 * (prop['psi'].log - state['psi'].log))
        accepted = prob > jax.random.uniform(rng_acc, prob.shape)
        if self.target_acceptance:
            acceptance = accepted.astype(int).sum().item() / accepted.shape[0]
            tau /= self.target_acceptance / max(acceptance, 0.05)
        state = {**state, 'age': state['age'] + 1}
        state = jax.tree_map(lambda xp, x: vec_where(accepted, xp, x), prop, state)
        return (state['r'], *((state['edges'],) if self.edge_builder else ())), {
            **state,
            'tau': tau,
        }


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
