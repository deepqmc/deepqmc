from functools import partial

import jax

PMAP_AXIS_NAME = 'device_axis'


def pmap(fn, axis_name=PMAP_AXIS_NAME, **kwargs):
    r"""Alias of jax.pmap, with default :data:`axis_name` value for convenience."""
    return jax.pmap(fn, axis_name, **kwargs)


def pmean(x, axis_name=PMAP_AXIS_NAME, **kwargs):
    r"""Alias of jax.lax.pmean, with default :data:`axis_name` value for convenience."""
    return jax.lax.pmean(x, axis_name, **kwargs)


def pmax(x, axis_name=PMAP_AXIS_NAME, **kwargs):
    r"""Alias of jax.lax.pmax, with default :data:`axis_name` value for convenience."""
    return jax.lax.pmax(x, axis_name, **kwargs)


def replicate_on_devices(pytree):
    r"""Replicate the input pytree on all devices.

    Tiles the input arrays to add a leading device axis. The data will be the same
    across all devices. The effect is analogous to calling
    :data:`jnp.repeat(input[None], jax.device_count(), 0)`, except that it also works
    for pytrees, and the output array will be sharded across the devices. Useful for
    replicating the same data across all devices.
    """
    return jax.device_put_replicated(pytree, devices=jax.devices())


@jax.pmap
def broadcast_to_devices(pytree):
    r"""Broadcast an array stored on a single device to all devices.

    The input array must already have the properly sized leading device axis
    (:data:`input.shape[0] == jax.device_count()`). Useful for broadcasting data
    that differs across devices to the devices.
    """
    return pytree


def select_one_device(pytree, idx=0):
    r"""Select one entry from the device axis.

    Selects the a single entry from the device axis, resulting in an array that is
    stored only on a single device. Useful for getting data that is identical across
    devices to a single device. Can be thought of as an inverse of
    :class:`deepqmc.parallel.replicate_on_devices`.

    Args:
        pytree: the input pytree of arrays.
        idx: the index of the entry to select from the leading device axis.
    """
    return jax.tree_util.tree_map(lambda x: x[idx], pytree)


def split_rng_key_to_devices(rng):
    r"""Create and place a separate rng key on each device.

    Args:
        rng: a simple rng key stored on a single device.
    """
    rngs = jax.random.split(rng, jax.device_count())
    return broadcast_to_devices(rngs)


@partial(jax.pmap, static_broadcasted_argnums=1)
def split_on_devices(rng, num):
    r"""Call the :class:`jax.random.split` function on each device.

    Args:
        rng: rng key with a leading device axis, rng keys stored on each device.
        num (int): the number of output keys on each device.
    """
    return tuple(jax.random.split(rng, num))


def rng_iterator(rng):
    r"""Create an rng key iterator on each device.

    Args:
        rng: rng key with a leading device axis, rng keys stored on each device.
    """
    while True:
        rng_yield, rng = split_on_devices(rng, 2)
        yield rng_yield


def pexp_normalize_mean(x, axis_name=PMAP_AXIS_NAME):
    r"""Compute the normalized-mean exponential of the input across many devices."""
    x_max = pmax(x.max())
    exp = jax.numpy.exp(x - x_max)
    exp_mean = pmean(exp.mean())
    return exp / exp_mean


def all_device_mean(x, axis_name=PMAP_AXIS_NAME, **mean_kwargs):
    r"""Compute mean across all devices.

    Args:
        x: the input data stored on multiple devices.
        axis_name: optional, name of pmap-ed axis.
    """
    return pmean(jax.numpy.mean(x, **mean_kwargs), axis_name)


def all_device_median(x, axis_name=PMAP_AXIS_NAME):
    r"""Compute median across all devices.

    Args:
        x: the input data stored on multiple devices.
        axis_name: optional, name of pmap-ed axis.
    """
    return jax.numpy.median(jax.lax.all_gather(x, axis_name))


def all_device_quantile(x, quantile, axis_name=PMAP_AXIS_NAME):
    r"""Compute quantiles across all devices.

    Args:
        x: the input data stored on multiple devices.
        quantile: probability for the quantiles to compute.
        axis_name: optional, name of pmap-ed axis.
    """
    return jax.numpy.quantile(jax.lax.all_gather(x, axis_name), quantile)


@partial(jax.pmap, axis_name='gather_axis')
def pmap_all_gather(x):
    r"""Gather data from all devices.

    Includes it's own :data:`pmap` call inside.
    """
    return jax.lax.all_gather(x, 'gather_axis')


@partial(jax.pmap, axis_name='pmean_axis')
def pmap_pmean(x):
    r"""Gather data using pmean from all devices.

    Includes it's own :data:`pmap` call inside.
    """
    return jax.lax.pmean(x, 'pmean_axis')


def gather_electrons_on_one_device(pytree):
    r"""Gather electron sample type arrays on one device.

    Many arrays (e.g. local energies, wave function values, etc.) are of the shape
    :data:`[n_device, molecule_batch_size, electron_batch_size / n_device, ...]`. The
    total :data:`electron_batch_size` many samples are stored across the devices. This
    function gathers arrays like these from the devices, and merges the electron batch
    axes to arrive at the output shape
    :data:`[molecule_batch_size, electron_batch_size]`.

    Args:
        pytree: a pytree of arrays all with shape:
            :data:`[n_device, molecule_batch_size, electron_batch_size / n_device, ...]`

    Result:
        a pytree of arrays all with shape:
            :data:`[molecule_batch_size, electron_batch_size]`.
    """
    all_gathered = pmap_all_gather(pytree)
    on_one_device = select_one_device(all_gathered)
    return jax.tree_util.tree_map(
        lambda x: jax.numpy.moveaxis(x, 0, 1).reshape(x.shape[1], -1, *x.shape[3:]),
        on_one_device,
    )
