from collections import namedtuple
from math import ceil
from typing import Optional

import jax
import jax.numpy as jnp

__all__ = ()

EWMState = namedtuple(
    'EWMState', 'step params buffer mean var sqerr', defaults=6 * [None]  # type: ignore
)


def init_ewm(
    max_alpha: float = 0.999,
    decay_alpha: float = 10.0,
    window_size: Optional[int] = None,
):
    if window_size is None:
        window_size = ceil(decay_alpha * (1 / (1 - max_alpha) - 2))

    state = EWMState(
        step=0,
        params={
            'max_alpha': max_alpha,
            'decay_alpha': decay_alpha,
            'alpha': jnp.zeros(window_size),
        },
        buffer=jnp.zeros(window_size),
        mean=jnp.nan,
        var=jnp.nan,
        sqerr=jnp.nan,
    )

    @jax.jit
    def update(x, state):
        max_alpha, decay_alpha, alpha = (
            state.params['max_alpha'],
            state.params['decay_alpha'],
            state.params['alpha'],
        )
        if state.mean is None:
            state.params['alpha'] = state.params['alpha'].at[0].set(1.0)
            return state._replace(
                buffer=state.buffer.at[0].set(x),
                step=0,
                mean=x,
                var=jnp.array(1.0),
                sqerr=jnp.array(1.0),
                median=jnp.array(1.0),
            )
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

        return state._replace(
            step=state.step + 1,
            buffer=buffer,
            mean=mean,
            var=variance.sum(),
            sqerr=(weights * variance).sum(),
        )

    return state, update


def init_multi_mol_multi_state_ewm(
    shape: tuple[int, ...],
    max_alpha: float = 0.999,
    decay_alpha: float = 10.0,
    window_size: Optional[int] = None,
):
    state, update = init_ewm(max_alpha, decay_alpha, window_size)

    def state_tree_map(fn, *state):
        return jax.tree_map(fn, *state, is_leaf=lambda x: isinstance(x, jax.Array))

    def vmapper(fn):
        for _ in range(len(shape)):
            fn = jax.vmap(fn)
        return fn

    def extend_state(state, shape):
        return state_tree_map(
            lambda x: jnp.broadcast_to(
                x, (*shape, *(x.shape if hasattr(x, 'shape') else ()))
            ),
            state,
        )

    def sub_state_getter(state, sub_idxs: Optional[jax.Array] = None):
        processed_sub_idxs = slice(None) if sub_idxs is None else sub_idxs
        return state_tree_map(lambda x: x[processed_sub_idxs], state)

    def sub_state_setter(state, state_update, sub_idxs: Optional[jax.Array] = None):
        processed_sub_idxs = slice(None) if sub_idxs is None else sub_idxs
        return state_tree_map(
            lambda x, y: x.at[processed_sub_idxs].set(y), state, state_update
        )

    def multi_update(x, state, sub_idxs: Optional[jax.Array] = None):
        sub_state = sub_state_getter(state, sub_idxs)
        new_sub_state = vmapper(update)(x, sub_state)
        return sub_state_setter(state, new_sub_state, sub_idxs)

    state = extend_state(state, shape)

    return state, multi_update
