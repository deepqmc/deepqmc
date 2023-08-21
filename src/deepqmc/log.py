import pickle
from collections import namedtuple
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tensorboard.summary
from jax.tree_util import tree_map

from .utils import gather_on_one_device, select_one_device

__all__ = ['CheckpointStore', 'H5LogTable', 'TensorboardMetricLogger']

Checkpoint = namedtuple('Checkpoint', 'step loss path')


class CheckpointStore:
    r"""Stores training checkpoints in the working directory.

    Args:
        workdir (str): path where checkpoints are stored.
        size (int): maximum number of checkpoints stored at any time.
        interval (int): number of steps between two checkpoints.
    """

    PATTERN = 'chkpt-{}.pt'

    def __init__(self, workdir, *, size=3, interval=1000):
        self.workdir = Path(workdir)
        for p in self.workdir.glob(self.PATTERN.format('*')):
            p.unlink()
        self.size = size
        self.interval = interval
        self.chkpts = []
        self.buffer = None

    def update(self, step, state, loss=jnp.inf):
        self.buffer = (step, state, loss)
        if not self.chkpts or (step >= self.interval + self.chkpts[-1].step):
            self.dump()
        while len(self.chkpts) > self.size:
            self.chkpts.pop(0).path.unlink()

    def dump(self):
        step, state, loss = self.buffer
        path = self.workdir / self.PATTERN.format(step)
        with path.open('wb') as f:
            pickle.dump((step, state), f)
        self.chkpts.append(Checkpoint(step, loss, path))

    def close(self):
        if self.buffer and not any(tree_map(lambda x: x.is_deleted(), self.buffer[1])):
            self.dump()
        # If the training crashes KFAC might have already freed the buffers and the
        # state can no longer be dumped. Preventing this by keeping a copy significantly
        # impacts the performance and is therefore omitted.

    @property
    def last(self):
        chkpt = self.chkpts[-1]
        with chkpt.path.open('rb') as f:
            return pickle.load(f)


class H5LogTable:
    r"""An interface for writing results to HDF5 files."""

    def __init__(self, group):
        self._group = group

    def __getitem__(self, label):
        return self._group[label] if label in self._group else []

    def resize(self, size):
        for ds in self._group.values():
            ds.resize(size, axis=0)

    # mimicking Pytables API
    @property
    def row(self):
        class Appender:
            def __setitem__(_, label, row):  # noqa: B902, N805
                if isinstance(row, np.ndarray):
                    shape = row.shape
                elif isinstance(row, jnp.ndarray):
                    shape = row.shape
                elif isinstance(row, (float, int)):
                    shape = ()
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


class TensorboardMetricLogger:
    r"""An interface for writing metrics to Tensorboard."""

    def __init__(self, workdir, n_mol, *, period):
        self.global_writer = tensorboard.summary.Writer(workdir)
        self.per_mol_writers = [
            tensorboard.summary.Writer(f'{workdir}/{i}') for i in range(n_mol)
        ]
        self.period = period

    def update(self, step, stats, single_device_stats=None, prefix=None):
        r"""Update tensorboard writer with a dictionary of scalar entries.

        Args:
            step (int): the step at which to add the new entries.
            stats (dict): a dictionary containing the scalar entries to add.
        """
        if step % self.period:
            return
        if stats:
            stats = gather_stats_on_one_device(stats)
        per_mol = stats.pop('per_mol', {})
        if single_device_stats:
            single_device_per_mol = single_device_stats.pop('per_mol', {})
            stats.update(single_device_stats)
            per_mol.update(single_device_per_mol)
        for k, v in per_mol.items():
            for i, writer in enumerate(self.per_mol_writers):
                if not (jnp.isnan(v[i]) or jnp.isinf(v[i])):
                    writer.add_scalar(f'{prefix}/{k}' if prefix else k, v[i], step)
        for k, v in stats.items():
            self.global_writer.add_scalar(f'{prefix}/{k}' if prefix else k, v, step)

    def close(self):
        self.global_writer.close()
        for writer in self.per_mol_writers:
            writer.close()


def gather_stats_on_one_device(stats):
    per_mol = stats.pop('per_mol', {})
    # Remaining of stats contains only global statistics e.g. param_norm
    # these are replicated (and thus identical) across all devices
    stats = select_one_device(stats)
    per_mol = gather_on_one_device(per_mol, gather_fn=jax.lax.pmean)
    return {**stats, 'per_mol': per_mol}
