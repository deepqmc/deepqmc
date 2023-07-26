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


def replicate_on_devices(pytree):
    return jax.device_put_replicated(pytree, devices=jax.devices())


def broadcast_to_devices(pytree):
    return jax.pmap(lambda x: x)(pytree)


def select_one_device(pytree, idx=0):
    return jax.tree_util.tree_map(lambda x: x[idx], pytree)


def gather_on_one_device(
    pytree, gather_fn=jax.lax.all_gather, flatten_device_axis=False
):
    all_gathered = jax.pmap(
        lambda x: gather_fn(x, 'gather_axis'), axis_name='gather_axis'
    )(pytree)
    on_one_device = select_one_device(all_gathered)
    if flatten_device_axis:
        on_one_device = jax.tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]), on_one_device
        )
    return on_one_device


def split_rng_key_to_devices(rng):
    rngs = jax.random.split(rng, jax.device_count())
    return broadcast_to_devices(rngs)


def rng_iterator(rng):
    while True:
        rng_yield, rng = jax.pmap(lambda key: tuple(jax.random.split(key)))(rng)
        yield rng_yield
