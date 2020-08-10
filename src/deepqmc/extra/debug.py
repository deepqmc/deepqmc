import time
from collections import UserDict, defaultdict
from contextlib import contextmanager
from functools import wraps

import numpy as np
import torch

__all__ = ()


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

    def result(self, val):
        return val


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


@contextmanager
def timer():
    now = np.array(time.time())
    try:
        yield now
    finally:
        now[...] = time.time() - now


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


class DebugLogTable:
    def __init__(self):
        self._data = defaultdict(list)

    def __getitem__(self, label):
        return self._data[label]

    @property
    def row(self):
        class Appender:
            def __setitem__(_, label, row):  # noqa: B902, N805
                self._data[label].append(row)

        return Appender()
