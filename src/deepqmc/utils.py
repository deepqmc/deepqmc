import time
from collections import UserDict
from contextlib import contextmanager
from datetime import datetime
from functools import wraps

import numpy as np
import torch

from .errors import DeepQMCError

__all__ = ()


def get_flat_mesh(bounds, npts, device=None):
    edges = [torch.linspace(*b, n, device=device) for b, n in zip(bounds, npts)]
    grids = torch.meshgrid(*edges)
    return torch.stack(grids).flatten(start_dim=1).t(), edges


def integrate_on_mesh(func, bounds, density=0.02):
    ns_pts = [int((bs[1] - bs[0]) / density) for bs in bounds]
    vol = np.array([bs[1] - bs[0] for bs in bounds]).prod()
    mesh = get_flat_mesh(bounds, ns_pts)[0]
    return sum(func(x).sum() for x in mesh.chunk(100)) * (vol / mesh.shape[0])


# TODO refactor as NestedDict
class DebugContainer(UserDict):
    def __init__(self):
        super().__init__()
        self._levels = []

    @contextmanager
    def cd(self, label):
        self._levels.append(label)
        try:
            yield
        finally:
            assert label == self._levels.pop()

    def _getkey(self, key):
        if isinstance(key, int) and not self._levels:
            return key
        return '.'.join([*self._levels, str(key)])

    def __getitem__(self, key):
        key = self._getkey(key)
        try:
            val = super().__getitem__(key)
        except KeyError:
            val = self.__class__()
            self.__setitem__(key, val)
        return val

    def __setitem__(self, key, val):
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu()
        super().__setitem__(self._getkey(key), val)

    def result(self, val):
        super().__setitem__('.'.join(self._levels), val)
        return val


class _NullDebug(DebugContainer):
    def __setitem__(self, key, val):
        pass


NULL_DEBUG = _NullDebug()


def debugged(func, label):
    @wraps(func)
    def wrapped(*args, **kwargs):
        debug = DebugContainer()
        func(*args, **kwargs, debug=debug)
        return debug[label]

    return wrapped


class Debuggable:
    def debug(self, label):
        return debugged(self, label)


def batch_eval(func, batches, *args, **kwargs):
    return torch.cat([func(batch, *args, **kwargs) for batch in batches])


def batch_eval_tuple(func, batches, *args, **kwargs):
    results = list(zip(*(func(batch, *args, **kwargs) for batch in batches)))
    return tuple(torch.cat(result) for result in results)


@contextmanager
def timer():
    now = np.array(time.time())
    try:
        yield now
    finally:
        now[...] = time.time() - now


def now():
    return datetime.now().isoformat(timespec='seconds')


def expand_1d(r, x, k, i):
    rs = r.repeat(len(x), 1, 1)
    rs[:, k, i] += x
    return rs


class NestedDict(dict):
    def __init__(self, dct=None):
        super().__init__()
        if dct:
            self.update(dct)

    def _split_key(self, key):
        key, *nested_key = key.split('.', 1)
        return (key, nested_key[0]) if nested_key else (key, None)

    def __getitem__(self, key):
        key, nested_key = self._split_key(key)
        try:
            val = super().__getitem__(key)
        except KeyError:
            val = NestedDict()
            super().__setitem__(key, val)
        if nested_key:
            return val[nested_key]
        return val

    def __setitem__(self, key, val):
        key, nested_key = self._split_key(key)
        if nested_key:
            self[key][nested_key] = val
        else:
            super().__setitem__(key, val)

    def __delitem__(self, key):
        key, nested_key = self._split_key(key)
        if nested_key:
            del super().__getitem__(key)[nested_key]
        else:
            super().__delitem__(key)

    def update(self, other):
        for key, val in other.items():
            if isinstance(val, dict):
                if not isinstance(self[key], NestedDict):
                    if isinstance(self[key], dict):
                        self[key] = NestedDict(self[key])
                    else:
                        self[key] = NestedDict()
                super().__getitem__(key).update(val)
            else:
                super().__setitem__(key, val)


def estimate_optimal_batch_size_cuda(
    func, test_batch_sizes, mem_margin=0.9, max_memory=None,
):
    assert len(test_batch_sizes) >= 4
    test_batch_sizes = torch.as_tensor(test_batch_sizes).float()
    mem = []
    for size in test_batch_sizes.int():
        torch.cuda.reset_max_memory_allocated()
        func(size.item())
        mem.append(torch.cuda.max_memory_allocated() / 1e6)
    mem = torch.tensor(mem)
    delta = (mem[1:] - mem[:-1]) / (test_batch_sizes[1:] - test_batch_sizes[:-1])
    delta = delta[1:]  # throw away first try
    assert (delta > 0).all()
    memory_per_batch = delta.mean() / mem_margin
    if torch.sqrt(delta.var()) / memory_per_batch > 0.3:
        raise DeepQMCError(
            'Inconsistent estimation of GPU memory per batch. '
            'Try specifying large test_batch_sizes.'
        )
    if max_memory is None:
        import subprocess

        memory_total = torch.cuda.get_device_properties(0).total_memory * 9.5367e-7
        sp = subprocess.Popen(
            ['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        memory_in_use = int(
            str(sp.communicate()).split('Used GPU Memory         : ')[1].split('MiB')[0]
        )
        max_memory = memory_total - memory_in_use
    return int(max_memory / memory_per_batch)
