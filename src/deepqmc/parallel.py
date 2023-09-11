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
    for pytrees, and the output array will be sharded accross the devices. Useful for
    replicating the same data across all devices.
    """
    return jax.device_put_replicated(pytree, devices=jax.devices())


def broadcast_to_devices(pytree):
    r"""Broadcast an array stored on a single device to all devices.

    The input array must already have the properly sized leading device axis
    (:data:`input.shape[0] == jax.device_count()`). Useful for broadcasting data
    that differs across devices to the devices.
    """
    return jax.pmap(lambda x: x)(pytree)


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


def gather_on_one_device(
    pytree, gather_fn=jax.lax.all_gather, flatten_device_axis=False
):
    r"""Gather data stored across devices to a single device.

    Useful for collecting data that differs across devices to a single device.
    Can be though of as an inverse of
    :class:`deepqmc.parallel.broadcast_to_devices`.

    Args:
        pytree: the input pytree of arrays to gather.
        gather_fn: default: :data:`jax.lax.all_gather`, the function used to
            gather/accumulate data. The default :data:`all_gather` results in
            the data being simply gathered. Passing e.g. :data:`jax.lax.pmean`
            would instead result in taking the mean across devices.
        flatten_device_axis: defaut: :data:`False`, calling
            :data:`jax.lax.all_gather` results in an output array that has a
            leading device axis (but all entries of this output are nonetheless
            stored on one device). If :data:`True`, this axis is flattened into
            the next axis.
    """
    all_gathered = jax.pmap(
        lambda x: gather_fn(x, PMAP_AXIS_NAME), axis_name=PMAP_AXIS_NAME
    )(pytree)
    on_one_device = select_one_device(all_gathered)
    if flatten_device_axis:
        on_one_device = jax.tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]), on_one_device
        )
    return on_one_device


def split_rng_key_to_devices(rng):
    r"""Create and place a separate rng key on each device.

    Args:
        rng: a simple rng key stored on a single device.
    """
    rngs = jax.random.split(rng, jax.device_count())
    return broadcast_to_devices(rngs)


def split_on_devices(rng, num=2):
    r"""Call the :class:`jax.random.split` function on each device.

    Args:
        rng: rng key with a leading device axis, rng keys stored on each device.
        num (int): the number of ouput keys on each device.
    """
    return jax.pmap(lambda key: tuple(jax.random.split(key, num)))(rng)


def rng_iterator(rng):
    r"""Create an rng key iterator on each device.

    Args:
        rng: rng key with a leading device axis, rng keys stored on each device.
    """
    while True:
        rng_yield, rng = split_on_devices(rng)
        yield rng_yield


def pexp_normalize_mean(x, axis_name=PMAP_AXIS_NAME):
    r"""Compute the normalized-mean exponential of the input across many devices."""
    x_max = pmax(x.max())
    exp = jax.numpy.exp(x - x_max)
    exp_mean = pmean(exp.mean())
    return exp / exp_mean


def all_device_mean(x, axis_name=PMAP_AXIS_NAME):
    r"""Compute mean across all devices.

    Args:
        x: the input data stored on multiple devices.
        axis_name: optional, name of pmap-ed axis.
    """
    return pmean(jax.numpy.mean(x), axis_name)


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
