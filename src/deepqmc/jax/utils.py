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


@jax.jit
@jax.vmap
def vec_where(cond, x, y):
    return jnp.where(cond, x, y)
