import pickle
from collections import namedtuple
from copy import deepcopy
from pathlib import Path

import jax.numpy as jnp
import numpy as np

Checkpoint = namedtuple('Checkpoint', 'step loss path')


class CheckpointStore:
    r"""Stores training checkpoints in the working directory.

    Args:
        workdir (str): path where checkpoints are stored.
        size (int): maximum number of checkpoints stored at any time.
        min_interval (str): minimum number of steps between two checkpoints.
        threshold (float): treshold for decrease in criterion for new checkpoint.
    """

    PATTERN = 'chkpt-{}.pt'

    def __init__(self, workdir, *, size=3, min_interval=100, threshold=0.95):
        self.workdir = Path(workdir)
        for p in self.workdir.glob(self.PATTERN.format('*')):
            p.unlink()
        self.size = size
        self.min_interval = min_interval
        self.threshold = threshold
        self.chkpts = []
        self.buffer = None

    def update(self, step, state, loss=jnp.inf):
        self.buffer = (step, loss, deepcopy(state))
        if step > self.min_interval + (self.chkpts[-1].step if self.chkpts else 0) and (
            loss <= self.threshold * (self.chkpts[-1].loss if self.chkpts else jnp.inf)
        ):
            self.dump(step, state, loss)

    def dump(self, step, state, loss=jnp.inf):
        path = self.workdir / self.PATTERN.format(step)
        with path.open('wb') as f:
            pickle.dump((step, state), f)
        self.chkpts.append(Checkpoint(step, loss, path))
        while len(self.chkpts) > self.size:
            self.chkpts.pop(0).path.unlink()

    def close(self):
        if self.buffer is not None:
            self.dump(*self.buffer)

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


def update_tensorboard_writer(writer, step, stats, prefix=None):
    r"""Update a tensorboard writer with a dictionary of scalar entries.

    Args:
        writer: the tensorboard writer.
        step (int): the step at which to add the new entries.
        stats (dict): a dictionary containing the scalar entries to add.
    """
    for k, v in stats.items():
        writer.add_scalar(f'{prefix}/{k}' if prefix else k, v, step)
