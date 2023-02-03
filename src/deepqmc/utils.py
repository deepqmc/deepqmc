from typing import Sequence

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
                wf_state = (
                    [st['wf'] for st in smpl_state]
                    if isinstance(smpl_state, Sequence)
                    else smpl_state['wf']
                )
                wf_state, overflow = state_callback(wf_state)
                if overflow:
                    smpl_state = (
                        [
                            {**prev, 'wf': st}
                            for prev, st in zip(smpl_state_prev, wf_state)
                        ]
                        if isinstance(smpl_state, Sequence)
                        else {**smpl_state_prev, 'wf': wf_state}
                    )
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


def argmax_random_choice(rng, x):
    mask = x == x.max()
    return jax.random.choice(rng, jnp.arange(len(x))[mask])


def segment_nanmean(data, segment_ids, num_segments=None):
    mask = ~jnp.isnan(data)
    data_zero = jnp.where(mask, data, 0)
    nanmean = ops.segment_sum(data_zero, segment_ids, num_segments) / mask.sum()
    return nanmean


def segment_nanstd(data, segment_ids, num_segments=None):
    mask = ~jnp.isnan(data)
    nanmean = segment_nanmean(data, segment_ids, num_segments)
    nanstd = jnp.where(mask, (nanmean[segment_ids] - data) ** 2, 0)
    nanstd = jnp.sqrt(ops.segment_sum(nanstd, segment_ids, num_segments) / mask.sum())
    return nanstd


def per_config_stats(n_configs, data, config_idx, prefix):
    mean = segment_nanmean(data, config_idx, n_configs)
    std = segment_nanstd(data, config_idx, n_configs)
    mask = ~jnp.isnan(data)
    minimum = ops.segment_min(jnp.where(mask, data, jnp.inf), config_idx, n_configs)
    maximum = ops.segment_max(jnp.where(mask, data, -jnp.inf), config_idx, n_configs)
    return {
        f'{prefix}/mean': mean,
        f'{prefix}/std': std,
        f'{prefix}/max': maximum,
        f'{prefix}/min': minimum,
    }


@jax.vmap
def sph2cart(sph, r=1):
    # This function transforms from spherical to cartesian coordinates.
    theta = sph[0]
    phi = sph[1]
    rsin_theta = r * jnp.sin(theta)
    x = rsin_theta * jnp.cos(phi)
    y = rsin_theta * jnp.sin(phi)
    z = r * jnp.cos(theta)
    return jnp.array([x, y, z])


def rot_y(theta):
    # returns the rotation matrix about y-axis by angle theta
    return jnp.array(
        [
            [jnp.cos(theta), jnp.zeros_like(theta), jnp.sin(theta)],
            [jnp.zeros_like(theta), jnp.ones_like(theta), jnp.zeros_like(theta)],
            [-jnp.sin(theta), jnp.zeros_like(theta), jnp.cos(theta)],
        ]
    )


def rot_z(phi):
    # returns the rotation matrix about z-axis by angle phi
    return jnp.array(
        [
            [jnp.cos(phi), -jnp.sin(phi), jnp.zeros_like(phi)],
            [jnp.sin(phi), jnp.cos(phi), jnp.zeros_like(phi)],
            [jnp.zeros_like(phi), jnp.zeros_like(phi), jnp.ones_like(phi)],
        ]
    )


def pad_list_of_3D_arrays_to_one_array(list_of_arrays):
    shapes = [jnp.asarray(arr).shape for arr in list_of_arrays]
    target_shape = jnp.max(jnp.array(shapes), axis=0)
    padded_arrays = [
        jnp.pad(
            array,
            [(0, target_shape[i] - array.shape[i]) for i in range(3)],
            mode='constant',
        )
        for array in list_of_arrays
    ]
    return jnp.array(padded_arrays)
