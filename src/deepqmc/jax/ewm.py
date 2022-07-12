from collections import namedtuple

import jax
import jax.numpy as jnp

__all__ = ()

EWMState = namedtuple('EWMState', 'params step mean var sqerr', defaults=4 * [None])


@jax.jit
def ewm(x=None, state=None, max_alpha=0.999, decay_alpha=10):
    if x is None:
        return EWMState({'max_alpha': max_alpha, 'decay_alpha': decay_alpha})
    if state.mean is None:
        return state._replace(step=0, mean=x, var=0, sqerr=0)
    p = state.params
    a = jnp.minimum(p['max_alpha'], 1 - 1 / (2 + state.step / p['decay_alpha']))
    return state._replace(
        step=state.step + 1,
        mean=(1 - a) * x + a * state.mean,
        var=(1 - a) * (x - state.mean) ** 2 + a * state.var,
        sqerr=(1 - a) ** 2 * state.var + a**2 * state.sqerr,
    )
