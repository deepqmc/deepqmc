import os
from collections.abc import Generator
from functools import partial
from typing import Optional, TypeVar

import jax
from jax._src.distributed import initialize
from jax.experimental.multihost_utils import broadcast_one_to_all

from .types import KeyArray

PMAP_AXIS_NAME = 'device_axis'

T = TypeVar('T')


def get_process_count() -> Optional[int]:
    r"""Get the number of processes in the current run.

    Detecting multiple processes only implemented for SLURM.
    """
    process_count = os.getenv('SLURM_NTASKS')
    if process_count is not None:
        return int(process_count)
    return None


def get_process_index() -> Optional[int]:
    r"""Get the process index of the current process.

    Detecting multiple processes only implemented for SLURM.
    """
    process_index = os.getenv('SLURM_PROCID')
    if process_index is not None:
        return int(process_index)
    return None


def maybe_init_multi_host():
    r"""Initialize multi-host training if multiple processes are detected.

    Detecting multiple processes only implemented for SLURM.
    """
    process_count = get_process_count()
    process_id = get_process_index()

    if process_count is not None and process_id is not None and int(process_count) > 1:
        cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
        assert cuda_visible_devices is not None
        initialize(
            num_processes=int(process_count),
            process_id=int(process_id),
            local_device_ids=[int(i) for i in cuda_visible_devices.split(',')],
        )


def pmap(fn, axis_name=PMAP_AXIS_NAME, **kwargs):
    r"""Alias of jax.pmap, with default :data:`axis_name` value for convenience."""
    return jax.pmap(fn, axis_name, **kwargs)


def pmean(x, axis_name=PMAP_AXIS_NAME, **kwargs):
    r"""Alias of jax.lax.pmean, with default :data:`axis_name` value for convenience."""
    return jax.lax.pmean(x, axis_name, **kwargs)


def pmax(x, axis_name=PMAP_AXIS_NAME, **kwargs):
    r"""Alias of jax.lax.pmax, with default :data:`axis_name` value for convenience."""
    return jax.lax.pmax(x, axis_name, **kwargs)


def pmin(x, axis_name=PMAP_AXIS_NAME, **kwargs):
    r"""Alias of jax.lax.pmin, with default :data:`axis_name` value for convenience."""
    return jax.lax.pmin(x, axis_name, **kwargs)


def replicate_on_devices(pytree, globally=False):
    r"""Replicate the input pytree on all devices.

    Tiles the input arrays to add a leading device axis. The data will be the same
    across all devices. The effect is analogous to calling
    :data:`jnp.repeat(input[None], jax.device_count(), 0)`, except that it also works
    for pytrees, and the output array will be sharded across the devices. Useful for
    replicating the same data across all devices.
    """
    pytree = jax.device_put_replicated(pytree, devices=jax.local_devices())
    if globally:
        # broadcast_on_to_all returns numpy arrays for some reason
        pytree = jax.tree_util.tree_map(
            jax.numpy.asarray,
            broadcast_one_to_all(pytree),
        )
    return pytree


@jax.pmap
def broadcast_to_devices(pytree: T) -> T:
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
    rngs = jax.random.split(rng, jax.local_device_count())
    return broadcast_to_devices(rngs)


def align_rng_key_across_devices(rng):
    r"""Aligns rng keys on multiple devices.

    Args:
        rng: the same rng key stored on each single device.
    """
    return pmax(rng)


@partial(jax.pmap, static_broadcasted_argnums=1)
def split_on_devices(rng, num):
    r"""Call the :class:`jax.random.split` function on each device.

    Args:
        rng: rng key with a leading device axis, rng keys stored on each device.
        num (int): the number of output keys on each device.
    """
    return tuple(jax.random.split(rng, num))


def rng_iterator(rng: KeyArray) -> Generator[KeyArray, None, None]:
    r"""Create an rng key iterator on each device.

    Args:
        rng: rng key with a leading device axis, rng keys stored on each device.
    """
    while True:
        rng_yield, rng = split_on_devices(rng, 2)
        yield rng_yield


def pexp_normalize_mean(x, axis_name=PMAP_AXIS_NAME):
    r"""Compute the normalized-mean exponential of the input across many devices."""
    x_max = pmax(x.max(), axis_name)
    exp = jax.numpy.exp(x - x_max)
    exp_mean = pmean(exp.mean(), axis_name)
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


def all_device_min(x, axis_name=PMAP_AXIS_NAME, **mean_kwargs):
    r"""Compute min across all devices.

    Args:
        x: the input data stored on multiple devices.
        axis_name: optional, name of pmap-ed axis.
    """
    return pmin(jax.numpy.min(x, **mean_kwargs), axis_name)


def all_device_max(x, axis_name=PMAP_AXIS_NAME, **mean_kwargs):
    r"""Compute max across all devices.

    Args:
        x: the input data stored on multiple devices.
        axis_name: optional, name of pmap-ed axis.
    """
    return pmax(jax.numpy.max(x, **mean_kwargs), axis_name)


def all_device_std(x, axis_name=PMAP_AXIS_NAME, **mean_kwargs):
    r"""Compute mean across all devices.

    Args:
        x: the input data stored on multiple devices.
        axis_name: optional, name of pmap-ed axis.
    """
    first_mean_kwargs = mean_kwargs | {'keepdims': True}
    mean = pmean(jax.numpy.mean(x, **first_mean_kwargs), axis_name)
    var = pmean(jax.numpy.mean((x - mean) ** 2, **mean_kwargs), axis_name)
    return jax.numpy.sqrt(var)


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


def gather_electrons_on_one_device(pytree, electron_batch_axis=3):
    r"""Gather electron sample type arrays on one device.

    Many arrays (e.g. local energies, wave function values, etc.) are of the shape
    :data:`[n_device, ..., electron_batch_size / n_device, ...]`. The
    total :data:`electron_batch_size` many samples are stored across the devices. This
    function gathers arrays like these from the devices, and merges the electron batch
    axes to arrive at the output shape :data:`[..., electron_batch_size, ...]`. The most
    common usecase involves arrays of shape :data:`[n_device, molecule_batch_size,
    electronic_states, electron_batch_size / n_device, ...]` and hence the axis of the
    electron batch is 3. The electron_batch_axis argument can be used if the axis of the
    electron batch differs from the regular case.

    Args:
        pytree: a pytree of arrays all with shape:
            :data:`[n_device, ... , electron_batch_size / n_device, ...]`
        electron_batch_axis: the axis carrying the electron batch

    Result:
        a pytree of arrays all with shape:
            :data:`[..., electron_batch_size, ...]`.
    """
    all_gathered = pmap_all_gather(pytree)
    on_one_device = select_one_device(all_gathered)
    return jax.tree_util.tree_map(
        lambda x: jax.numpy.moveaxis(x, 0, electron_batch_axis - 1).reshape(
            *x.shape[1:electron_batch_axis], -1, *x.shape[electron_batch_axis + 1 :]
        ),
        on_one_device,
    )


def local_slice() -> slice:
    r"""Return a slice selecting the local devices from an array of all devices."""
    local_devices = jax.local_device_count()
    process_idx = jax.process_index()
    return slice(process_idx * local_devices, (process_idx + 1) * local_devices)


def scatter_electrons_to_devices(pytree: T) -> T:
    r"""Scatter electron sample type arrays across all devices.

    Can be thought of as an inverse of
    :class:`~deepqmc.parallel.gather_electrons_on_one_device`.

    Args:
        pytree: a pytree of arrays all with shape:
            :data:`[molecule_batch_size, electronic_states, electron_batch_size]`

    Result:
        a pytree of arrays all with shape:
            :data:`[n_device, molecule_batch_size, electronic_states,
            electron_batch_size / n_device, ...]`
    """
    reshaped_pytree: T = jax.tree_util.tree_map(
        lambda x: jax.numpy.moveaxis(
            x.reshape(*x.shape[:2], jax.device_count(), -1, *x.shape[3:]), 2, 0
        )[local_slice()],
        pytree,
    )
    return broadcast_to_devices(reshaped_pytree)
