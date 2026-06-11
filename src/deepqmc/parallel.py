import os
from collections.abc import Generator
from functools import partial
from typing import Optional, TypeVar

import jax
import numpy as np
from jax.experimental.multihost_utils import broadcast_one_to_all

from .types import KeyArray

PMAP_AXIS_NAME = 'device_axis'

T = TypeVar('T')


def get_process_count() -> Optional[int]:
    r"""Get the number of processes in the current run.

    Multiple processes are detected from the :data:`SLURM_NTASKS` (on SLURM) or
    :data:`DEEPQMC_NUM_PROCESSES` (manual launch) environment variables.
    """
    process_count = os.getenv('SLURM_NTASKS') or os.getenv('DEEPQMC_NUM_PROCESSES')
    if process_count is not None:
        return int(process_count)
    return None


def get_process_index() -> Optional[int]:
    r"""Get the process index of the current process.

    The process index is detected from the :data:`SLURM_PROCID` (on SLURM) or
    :data:`DEEPQMC_PROCESS_ID` (manual launch) environment variables.
    """
    process_index = os.getenv('SLURM_PROCID') or os.getenv('DEEPQMC_PROCESS_ID')
    if process_index is not None:
        return int(process_index)
    return None


def get_local_device_ids() -> Optional[list[int]]:
    r"""Get the ids of the devices the current process should use.

    If :data:`JAX_LOCAL_DEVICE_IDS` is set, defer to jax, which parses it inside
    :func:`jax.distributed.initialize`. Otherwise, if :data:`CUDA_VISIBLE_DEVICES`
    is set, the process uses all the devices visible to it. Note that the visible
    devices are renumbered from zero within the process, so the raw ids from
    :data:`CUDA_VISIBLE_DEVICES` must not be passed to jax. If neither variable is
    set, returning :data:`None` defers to jax's cluster auto-detection, which on
    SLURM and Open MPI assigns one GPU per process.
    """
    if os.getenv('JAX_LOCAL_DEVICE_IDS'):
        return None
    cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices:
        return list(range(len(cuda_visible_devices.split(','))))
    return None


def maybe_init_multi_host():
    r"""Initialize multi-host training if multiple processes are detected.

    On SLURM clusters multiple processes are detected automatically. Otherwise, a
    multi-process run can be configured manually through the
    :data:`JAX_COORDINATOR_ADDRESS` (read by jax), :data:`DEEPQMC_NUM_PROCESSES`
    and :data:`DEEPQMC_PROCESS_ID` environment variables.
    """
    process_count = get_process_count()
    process_id = get_process_index()

    if process_count is not None and process_id is not None and process_count > 1:
        # coordinator_address is read from JAX_COORDINATOR_ADDRESS or detected
        # from the SLURM environment inside jax.distributed.initialize
        jax.distributed.initialize(
            num_processes=process_count,
            process_id=process_id,
            local_device_ids=get_local_device_ids(),
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

    Args:
        pytree: the input pytree of arrays.
        globally: if :data:`True`, the data of process zero is first broadcast to all
            processes, guaranteeing identical data on all devices of all processes.
    """
    if globally and jax.process_count() > 1:
        pytree = broadcast_one_to_all(pytree)
    n_devices = jax.local_device_count()
    # go through host numpy arrays, the input may be committed to a single device,
    # in which case it could not be fed to the pmapped broadcast_to_devices directly
    return jax.tree_util.tree_map(
        lambda x: broadcast_to_devices(
            np.repeat(np.asarray(x)[None], n_devices, axis=0)
        ),
        pytree,
    )


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

    def select(x):
        if isinstance(x, jax.Array) and len(x.sharding.device_set) > 1:
            # eagerly indexing arrays sharded across multiple devices is not
            # supported in multi-process runs, fetch the addressable shard instead
            return x.addressable_data(idx)[0]
        return x[idx]

    return jax.tree_util.tree_map(select, pytree)


@pmap
def select_local_entries(pytree: T) -> T:
    r"""Select the entries belonging to the local devices from gathered arrays.

    The input arrays must have a leading axis of size :data:`jax.device_count()`
    holding identical data on each local device (e.g. outputs of
    :func:`jax.lax.all_gather`). Each device selects the entry corresponding to its
    global position, undoing the :data:`all_gather`. Equivalent to
    :data:`select_one_device(gathered)[local_slice()]`, but without leaving the
    devices.
    """
    idx = jax.lax.axis_index(PMAP_AXIS_NAME)
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
    device_count = jax.device_count()

    def scatter(x):
        assert x.shape[2] % device_count == 0, (
            f'Cannot scatter electron batch of size {x.shape[2]} evenly across'
            f' {device_count} devices. This can happen when restoring a checkpoint'
            ' with a different total number of devices than it was created with.'
        )
        return jax.numpy.moveaxis(
            x.reshape(*x.shape[:2], device_count, -1, *x.shape[3:]), 2, 0
        )[local_slice()]

    reshaped_pytree: T = jax.tree_util.tree_map(scatter, pytree)
    return broadcast_to_devices(reshaped_pytree)
