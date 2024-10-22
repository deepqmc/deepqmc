import logging
import os
import pickle
import re
import sys
from copy import deepcopy
from functools import partial
from itertools import product
from pathlib import Path
from typing import NamedTuple, Optional, Protocol, Union

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map
from tensorboardX import SummaryWriter

from .parallel import (
    gather_electrons_on_one_device,
    pmap_pmean,
    replicate_on_devices,
    scatter_electrons_to_devices,
    select_one_device,
)
from .types import Stats, TrainState
from .utils import flatten_dict, tree_any

__all__ = ['CheckpointStore', 'H5LogTable', 'TensorboardMetricLogger']
log = logging.getLogger(__name__)


class Checkpoint(NamedTuple):
    step: int
    loss: jax.typing.ArrayLike
    path: Path


def serialize_train_state(train_state: TrainState) -> TrainState:
    params, opt = select_one_device((train_state.params, train_state.opt))
    sampler_state = deepcopy(train_state.sampler)
    sampler_nuc, update_nuc_counter = select_one_device(
        (sampler_state.pop('nuc'), sampler_state.pop('update_nuc_counter'))
    )
    sampler_state['elec'] = gather_electrons_on_one_device(sampler_state['elec'])
    sampler_state['nuc'] = sampler_nuc
    sampler_state['update_nuc_counter'] = update_nuc_counter
    sampler_state['elec']['tau'] = sampler_state['elec']['tau'].mean(axis=-1)
    return TrainState(sampler_state, params, opt)


def deserialize_train_state(train_state: TrainState) -> TrainState:
    if train_state.sampler['elec'].get('r', None) is not None:
        if train_state.sampler['elec']['r'].ndim == 6:
            # Legacy checkpoints are already deserialized
            return train_state
    if train_state.sampler['elec'].get('tau', None) is not None:
        if train_state.sampler['elec']['tau'].ndim == 3:
            # up to and including 147d4feb tau was not averaged over devices
            train_state.sampler['elec']['tau'] = train_state.sampler['elec'][
                'tau'
            ].mean(axis=-1)
    train_state.sampler['elec']['tau'] = jnp.repeat(
        train_state.sampler['elec']['tau'][..., None], jax.device_count(), axis=-1
    )
    params, opt = replicate_on_devices((train_state.params, train_state.opt))
    sampler = train_state.sampler
    sampler['elec'] = scatter_electrons_to_devices(sampler['elec'])
    sampler['elec']['tau'] = jnp.squeeze(sampler['elec']['tau'], axis=-1)
    sampler['nuc'], sampler['update_nuc_counter'] = replicate_on_devices(
        (sampler['nuc'], sampler['update_nuc_counter'])
    )
    return TrainState(sampler, params, opt)


class CheckpointStore:
    r"""Stores training checkpoints in the working directory.

    Args:
        workdir (str): path where checkpoints are stored.
        size (int): maximum number of checkpoints stored at any time.
        interval (int): number of steps between two checkpoints.
    """

    PATTERN = 'chkpt-{}.pt'

    def __init__(self, workdir: str, *, size: int = sys.maxsize, interval: int = 1000):
        self.workdir = Path(workdir)
        for p in self.workdir.glob(self.PATTERN.format('*')):
            p.unlink()
        self.size = size
        self.interval = interval
        self.chkpts: list[Checkpoint] = []
        self.buffer: Union[
            tuple[None, None, None], tuple[int, TrainState, jax.typing.ArrayLike]
        ] = (None, None, None)

    def update(
        self, step: int, state: TrainState, loss: jax.typing.ArrayLike = jnp.inf
    ):
        self.buffer = (step, state, loss)
        if not self.chkpts or (step >= self.interval + self.chkpts[-1].step):
            self.dump()
        while len(self.chkpts) > self.size:
            self.chkpts.pop(1).path.unlink()
            # We pop the first checkpoint to always keep the initial checkpoint
            # preceding the variational training.

    def dump(self):
        step, state, loss = self.buffer
        assert (
            isinstance(state, TrainState)
            and isinstance(step, int)
            and isinstance(loss, jax.typing.ArrayLike)
        )
        path = self.workdir / self.PATTERN.format(step)
        with path.open('wb') as f:
            pickle.dump((step, serialize_train_state(state)), f)
        self.chkpts.append(Checkpoint(step, loss, path))

    @staticmethod
    def load(path: Path) -> tuple[int, TrainState]:
        with open(path, 'rb') as f:
            step, state = pickle.load(f)
        return step, deserialize_train_state(state)

    def close(self):
        if all(self.buffer) and not tree_any(
            tree_map(lambda x: x.is_deleted(), self.buffer[1])
        ):
            self.dump()
        # If the training crashes KFAC might have already freed the buffers and the
        # state can no longer be dumped. Preventing this by keeping a copy significantly
        # impacts the performance and is therefore omitted.

    @property
    def last(self) -> tuple[int, TrainState]:
        chkpt = self.chkpts[-1]
        return self.load(chkpt.path)

    @classmethod
    def extract_step_from_filename(cls, filename: str) -> int:
        match = re.search(cls.PATTERN.format(r'(\d+)'), filename)
        if match is None:
            raise ValueError(f'Invalid checkpoint filename {filename}.')
        return int(match.groups()[0])


def resize_if_dataset(size: int, name: str, obj: Union[h5py.Dataset, h5py.Group]):
    r"""Resize dataset objects of HDF5 files.

    A ``partial`` of this function can be used as the visitor function argument of
    ``h5py.File.visititems``.
    """
    if isinstance(obj, h5py.Dataset):
        obj.resize(size, axis=0)


class H5LogTable:
    r"""An interface for writing results to HDF5 files."""

    def __init__(self, group):
        self._group = group

    def __getitem__(self, label):
        return self._group[label] if label in self._group else []

    def resize(self, size: int):
        self._group.visititems(partial(resize_if_dataset, size))

    # mimicking Pytables API
    @property
    def row(self):
        class Appender:
            def __setitem__(_self, label: str, row):  # noqa: B902, N805
                if isinstance(row, np.ndarray):
                    shape = row.shape
                elif isinstance(row, jax.Array):
                    shape = row.shape
                elif isinstance(row, (float, int)):
                    shape = ()
                else:
                    raise ValueError(f'Cannot append row of type {type(row)}.')
                if label not in self._group:
                    if isinstance(row, np.ndarray):
                        dtype = row.dtype
                    elif isinstance(row, float):
                        dtype = float
                    else:
                        dtype = None
                    self._group.create_dataset(
                        label, (0, *shape), maxshape=(None, *shape), dtype=dtype
                    )
                ds = self._group[label]
                ds.resize(ds.shape[0] + 1, axis=0)
                ds[-1, ...] = row

        return Appender()


class H5Logger:
    def __init__(
        self,
        workdir: str,
        observable_names: Optional[list[str]] = None,
        *,
        keys_whitelist: Optional[list[str]] = None,
        init_step: int = 0,
        aux_data: Optional[dict] = None,
    ):
        self.keys_whitelist = (
            keys_whitelist if keys_whitelist is not None else ['local_energy']
        ) + (observable_names or [])
        self.h5file = h5py.File(os.path.join(workdir, 'result.h5'), 'a', libver='v110')
        self.h5file.swmr_mode = True
        for k, v in (aux_data or {}).items():
            self.h5file.attrs.create(k, v)
        self.table = H5LogTable(self.h5file)
        self.table.resize(init_step)
        self.flush()

    def update(self, single_device_data: Stats):
        data = flatten_dict(single_device_data)
        data_filtered = {
            key: value
            for key, value in data.items()
            if any(
                whitelisted_pharse in key for whitelisted_pharse in self.keys_whitelist
            )
        }
        for k, v in data_filtered.items():
            self.write(k, v)

        self.flush()

    def write(self, key: str, data: jax.Array):
        self.table.row[key] = data

    def flush(self):
        self.h5file.flush()

    def close(self):
        self.h5file.close()


class MetricLogger(Protocol):
    r"""Protocol for a general MetricLogger."""

    def __init__(self, workdir: str, n_mol: int):
        r"""Instantiates an object that loggs training metrics to the workdir."""
        ...

    def update(
        self,
        step: int,
        single_device_stat: Stats,
        multi_device_stats: Stats,
        mol_idxs: jax.Array,
        prefix: Optional[str] = None,
    ):
        r"""Update the MetricLoger with single and multi device stats.

        Args:
            step (int): the step at which to add the new entries.
            single_device_stats (dict): a dictionary containing the entries to add,
                that are on a single device.
            multi_device_stats (dict): a dictionary containing the entries to add,
                that are stored over multiple devices.
            mol_idxs (Array[int]): indices of molecules considered in the given step.
            prefix (str): optional, an optional prefix to append to the keys.
        """
        ...

    def add_full_dataset_scalars(self, stats: Stats, prefix: Optional[str]): ...

    def write_in_full_dataset_format(
        self, step: int, stats: Stats, mol_idxs: jax.Array, prefix: Optional[str]
    ): ...

    def add_batch_scalars(self, stats: Stats, prefix: Optional[str]): ...

    def write_in_batch_format(
        self, step, stats: Stats, mol_idxs: jax.Array, prefix: Optional[str]
    ): ...

    def close(self): ...


class TensorboardMetricLogger:
    r"""An interface for writing metrics to Tensorboard."""

    def __init__(self, workdir: str, n_mol: int, *, max_queue: int = 10):
        self.writer = SummaryWriter(workdir, max_queue=max_queue)
        self.n_mol = n_mol
        self.layout: dict = {}

    def update(
        self,
        step: int,
        single_device_stats: Stats,
        multi_device_stats: Stats,
        mol_idxs: jax.Array,
        prefix: Optional[str] = None,
    ):
        r"""Update tensorboard writer with a dictionary of entries.

        Args:
            step (int): the step at which to add the new entries.
            single_device_stats (dict): a dictionary containing the entries to add,
                that are on a single device.
            multi_device_stats (dict): a dictionary containing the entries to add,
                that are stored over multiple devices.
            mol_idxs (Array[int]): indices of molecules considered in the given step.
            prefix (~.typingOptional[str]): an optional prefix to append to the stat
                keys.
        """
        prefix = f'{prefix}/' if prefix else ''
        if multi_device_stats:
            all_mean = pmap_pmean(multi_device_stats)
            multi_device_stats = select_one_device(all_mean)
        stats = {**multi_device_stats, **single_device_stats}
        if self.n_mol <= 100:
            self.write_in_full_dataset_format(step, stats, mol_idxs, prefix)
        else:
            self.write_in_batch_format(step, stats, mol_idxs, prefix)

    def add_full_dataset_scalars(self, stats: Stats, prefix: Optional[str]):
        for k, v in stats.items():
            if v.ndim == 0:
                # Global statistics are not combined
                continue
            elif v.ndim == 1:
                # Combined molecule statistic
                keys = [f'{prefix}{k}/{i}' for i in range(self.n_mol)]
            elif v.ndim == 2:
                # Combined molecule and state statistic
                keys = [
                    f'{prefix}{k}/{i}/{j}'
                    for i, j in product(range(self.n_mol), range(v.shape[1]))
                ]
            elif v.ndim == 3:
                # Combined molecule and pairwise state statistic
                keys = [
                    f'{prefix}{k}/{i}/{j}-{l}'
                    for i, j, l in product(
                        range(self.n_mol), range(v.shape[1]), range(v.shape[2])
                    )
                ]
            else:
                log.warning(
                    f'Invalid dimension ({v.ndim}) for {k} (shape: {v.shape}), '
                    'excluding it from Tensorboard log.'
                )
                continue
            group = k.split('/')[0]
            self.layout[f'{prefix}{group}'] = {
                k: ['Multiline', keys],
                **self.layout.get(f'{prefix}{group}', {}),
            }
        self.writer.add_custom_scalars(self.layout)

    def write_in_full_dataset_format(
        self, step: int, stats: Stats, mol_idxs: jax.Array, prefix: Optional[str]
    ):
        if step == 0:
            self.add_full_dataset_scalars(stats, prefix)

        for k, v in stats.items():
            if v.ndim == 0:
                # Global statistic
                self.writer.add_scalar(f'{prefix}{k}', v, step)
            elif v.ndim == 1:
                # Per molecule statistic
                for i, v_i in zip(mol_idxs, v):
                    self.writer.add_scalar(f'{prefix}{k}/{i}', v_i, step)
            elif v.ndim == 2:
                # Per molecule per state statistic
                for i, v_i in zip(mol_idxs, v):
                    for j, v_ij in enumerate(v_i):
                        self.writer.add_scalar(f'{prefix}{k}/{i}/{j}', v_ij, step)
            elif v.ndim == 3:
                assert v.shape[1] == v.shape[2]
                # Per molecule per state pairwise statistic (upper triangular)
                for i, v_i in zip(mol_idxs, v):
                    for j, l in zip(*jnp.triu_indices(v.shape[2], k=1)):
                        self.writer.add_scalar(
                            f'{prefix}{k}/{i}/{l}-{j}', v_i[j, l], step
                        )

    def add_batch_scalars(self, stats: Stats, prefix: Optional[str]):
        for k, v in stats.items():
            if v.ndim == 0:
                # Global statistics are not combined
                continue
            elif v.ndim == 1:
                # Per molecule statistics: mean and std over molecule batch
                keys = [f'{prefix}{k}/mean', f'{prefix}{k}/std']
            elif v.ndim == 2:
                # Per molecule per state statistics: mean and std over molecule batch
                keys = [f'{prefix}{k}/mean/{j}' for j in range(v.shape[1])] + [
                    f'{prefix}{k}/std/{j}' for j in range(v.shape[1])
                ]
            elif v.ndim == 3:
                # Per molecule per state pair statistics:
                # mean and std over molecule batch
                keys = [
                    f'{prefix}{k}/mean/{j}-{l}'
                    for j, l in product(range(v.shape[1]), range(v.shape[2]))
                ] + [
                    f'{prefix}{k}/std/{j}-{l}'
                    for j, l in product(range(v.shape[1]), range(v.shape[2]))
                ]
            else:
                log.warning(
                    f'Invalid dimension ({v.ndim}) for {k} (shape: {v.shape}), '
                    'excluding it from Tensorboard log.'
                )
                continue
            group = k.split('/')[0]
            self.layout[f'{prefix}{group}'] = {
                k: ['Multiline', keys],
                **self.layout.get(f'{prefix}{group}', {}),
            }
        self.writer.add_custom_scalars(self.layout)

    def write_in_batch_format(
        self, step: int, stats: Stats, mol_idxs: jax.Array, prefix: Optional[str]
    ):
        if step == 0:
            self.add_batch_scalars(stats, prefix)

        for k, v in stats.items():
            if v.ndim == 0:
                # Global statistic
                self.writer.add_scalar(f'{prefix}{k}', v, step)
            elif v.ndim == 1:
                # Per molecule statistic
                self.writer.add_scalar(f'{prefix}{k}/mean', v.mean(), step)
                self.writer.add_scalar(f'{prefix}{k}/std', v.std(), step)
            elif v.ndim == 2:
                # Per molecule per state statistic
                v_mean = v.mean(axis=0)
                v_std = v.std(axis=0)
                for j, (v_mean_j, v_std_j) in enumerate(zip(v_mean, v_std)):
                    self.writer.add_scalar(f'{prefix}{k}/mean/{j}', v_mean_j, step)
                    self.writer.add_scalar(f'{prefix}{k}/std/{j}', v_std_j, step)
            elif v.ndim == 3:
                assert v.shape[1] == v.shape[2]
                # Per molecule per state pairwise statistic (upper triangular)
                v_mean = v.mean(axis=0)
                v_std = v.std(axis=0)
                for j, l in zip(*jnp.triu_indices(v.shape[2], k=1)):
                    self.writer.add_scalar(
                        f'{prefix}{k}/mean/{l}-{j}', v_mean[j, l], step
                    )
                    self.writer.add_scalar(
                        f'{prefix}{k}/std/{l}-{j}', v_std[j, l], step
                    )

    def close(self):
        self.writer.close()
