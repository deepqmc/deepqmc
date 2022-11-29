import jax
import jax.numpy as jnp
from jax.random import uniform
from jax.scipy.special import gammaln

__all__ = ()


def flatten(x, start_axis=0):
    return x.reshape(*x.shape[:start_axis], -1)


def unflatten(x, axis, shape):
    if axis < 0:
        axis = len(x.shape) + axis
    begin = x.shape[:axis]
    end = x.shape[axis + 1 :]
    return x.reshape(*begin, *shape, *end)


def multinomial_resampling(rng, weights, n_samples=None):
    n = len(weights)
    n_samples = n_samples or n
    weights_normalized = weights / jnp.sum(weights)
    i, j = jnp.triu_indices(n)
    weights_cum = jnp.zeros((n, n)).at[i, j].set(weights_normalized[j]).sum(axis=-1)
    return n - 1 - (uniform(rng, (n_samples,))[:, None] > weights_cum).sum(axis=-1)


def factorial2(n):
    n = jnp.asarray(n)
    gamma = jnp.exp(gammaln(n / 2 + 1))
    factor = jnp.where(
        n % 2, jnp.power(2, n / 2 + 0.5) / jnp.sqrt(jnp.pi), jnp.power(2, n / 2)
    )
    return factor * gamma


def masked_mean(x, mask):
    x = jnp.where(mask, x, 0)
    return x.sum() / jnp.sum(mask)


def exp_normalize_mean(x):
    x_shifted = x - x.max()
    return jnp.exp(x_shifted) / jnp.exp(x_shifted).mean()


def triu_flat(x):
    i, j = jnp.triu_indices(x.shape[-1], 1)
    return x[..., i, j]


def tree_norm(x):
    return jax.tree_util.tree_reduce(lambda norm, x: norm + jnp.linalg.norm(x), x, 0)


def norm(rs, safe=False, axis=-1):
    eps = jnp.finfo(rs.dtype).eps
    return (
        jnp.sqrt(eps + (rs * rs).sum(axis=axis))
        if safe
        else jnp.linalg.norm(rs, axis=axis)
    )


def split_dict(dct, cond):
    included, excluded = {}, {}
    for k, v in dct.items():
        (included if cond(k) else excluded)[k] = v
    return included, excluded


def check_overflow(state_callback, func):
    def wrapper(rng, smpl_state, *args, **kwargs):
        while True:
            smpl_state, *other = func(
                rng, smpl_state_prev := smpl_state, *args, **kwargs
            )
            if state_callback:
                wf_state, overflow = state_callback(smpl_state['wf'])
                if overflow:
                    smpl_state = {**smpl_state_prev, 'wf': wf_state}
                    continue
            return smpl_state, *other

    return wrapper


def no_grad(func):
    def wrapper(*args, **kwargs):
        args = jax.tree_util.tree_map(jax.lax.stop_gradient, args)
        return func(*args, **kwargs)

    return wrapper


def InverseSchedule(init_value, decay_rate):
    return lambda n: init_value / (1 + n / decay_rate)
