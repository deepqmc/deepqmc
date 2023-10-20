import jax
import jax.numpy as jnp
from jax import ops
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


def triu_flat(x):
    i, j = jnp.triu_indices(x.shape[-1], 1)
    return x[..., i, j]


def tree_norm(x):
    return jax.tree_util.tree_reduce(lambda norm, x: norm + jnp.linalg.norm(x), x, 0)


def tree_any(x):
    return jax.tree_util.tree_reduce(lambda is_any, leaf: is_any or leaf, x, False)


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


def InverseSchedule(init_value, decay_rate):
    return lambda n: init_value / (1 + n / decay_rate)


def ConstantSchedule(value):
    return lambda n: value


def argmax_random_choice(rng, x):
    logits = jnp.where(x == x.max(), 0, -jnp.inf)
    return jax.random.categorical(rng, logits, shape=())


def segment_nanmean(data, segment_ids, num_segments=None):
    mask = ~jnp.isnan(data)
    counts = jnp.bincount(
        jnp.where(mask, segment_ids, num_segments), length=num_segments
    )
    data_zero = jnp.where(mask, data, 0)
    nanmean = ops.segment_sum(data_zero, segment_ids, num_segments) / counts
    return nanmean


def segment_nanstd(data, segment_ids, num_segments=None):
    mask = ~jnp.isnan(data)
    counts = jnp.bincount(
        jnp.where(mask, segment_ids, num_segments), length=num_segments
    )
    nanmean = segment_nanmean(data, segment_ids, num_segments)
    nanstd = jnp.where(mask, (nanmean[segment_ids] - data) ** 2, 0)
    nanstd = jnp.sqrt(ops.segment_sum(nanstd, segment_ids, num_segments) / counts)
    return nanstd


def per_mol_stats(n_mols, data, mol_idx, prefix, mean_only=False):
    mean = segment_nanmean(data, mol_idx, n_mols)
    if mean_only:
        return mean
    std = segment_nanstd(data, mol_idx, n_mols)
    mask = ~jnp.isnan(data)
    minimum = ops.segment_min(jnp.where(mask, data, jnp.inf), mol_idx, n_mols)
    maximum = ops.segment_max(jnp.where(mask, data, -jnp.inf), mol_idx, n_mols)
    return {
        f'{prefix}/mean': mean,
        f'{prefix}/std': std,
        f'{prefix}/max': maximum,
        f'{prefix}/min': minimum,
    }


def log_squeeze(x):
    sgn, x = jnp.sign(x), jnp.abs(x)
    return sgn * jnp.log1p((x + 1 / 2 * x**2 + x**3) / (1 + x**2))
