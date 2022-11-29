from collections import namedtuple
from math import ceil

import jax
import jax.numpy as jnp

__all__ = ()

EWMState = namedtuple(
    'EWMState', 'step params buffer mean var sqerr', defaults=6 * [None]
)


def init_ewm(
    max_alpha=0.999,
    decay_alpha=10,
    window_size=None,
    clip=1000.0,
    max_exclude_ratio=0.0005,
):
    if window_size is None:
        window_size = ceil(decay_alpha * (1 / (1 - max_alpha) - 2))
    state = EWMState(
        step=0,
        params={
            'clip': clip,
            'max_alpha': max_alpha,
            'decay_alpha': decay_alpha,
            'alpha': jnp.zeros(window_size),
        },
        buffer=jnp.zeros(window_size),
    )

    @jax.jit
    def update(E_loc, state):
        clip, max_alpha, decay_alpha, alpha = (
            state.params['clip'],
            state.params['max_alpha'],
            state.params['decay_alpha'],
            state.params['alpha'],
        )
        if state.mean is None:
            state.params['alpha'] = state.params['alpha'].at[0].set(1.0)
            x = jnp.nanmean(E_loc)
            return state._replace(
                buffer=state.buffer.at[0].set(x),
                step=0,
                mean=x,
                var=jnp.array(1.0),
                sqerr=jnp.array(1.0),
            )
        diff = jnp.abs(state.mean - E_loc)
        excluded = diff > clip * state.var
        E_loc = jnp.where(
            excluded,
            state.mean,
            E_loc,
        )
        clip *= (
            jnp.maximum(excluded.mean(), jnp.array(max_exclude_ratio))
            / max_exclude_ratio
        )
        x = jnp.nanmean(E_loc)
        buffer = jnp.concatenate([x[None], state.buffer[:-1]])
        alpha = jax.lax.cond(
            state.step + 1 >= len(alpha),
            lambda: alpha,
            lambda: jnp.concatenate(
                [
                    jnp.maximum(1 - max_alpha, 1 / (2 + state.step / decay_alpha))[
                        None
                    ],
                    alpha[:-1],
                ]
            ),
        )
        beta = jnp.concatenate([jnp.array([1.0]), jnp.cumprod(1 - alpha[:-1])])
        weights = alpha * beta
        mean = (weights * buffer).sum()
        variance = weights * (buffer - mean) ** 2
        state.params['alpha'] = alpha
        state.params['clip'] = clip
        return state._replace(
            step=state.step + 1,
            buffer=buffer,
            mean=mean,
            var=variance.sum(),
            sqerr=(weights * variance).sum(),
        )

    return state, update
