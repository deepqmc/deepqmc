import inspect

import numpy as np
import tomlkit
from tomlkit.items import Comment, Trivia

__all__ = ()


class H5LogTable:
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
                        label, (0, *shape), maxshape=(None, *shape), dtype=dtype,
                    )
                ds = self._group[label]
                ds.resize(ds.shape[0] + 1, axis=0)
                ds[-1, ...] = row

        return Appender()


class _EnergyOffset:
    value = None

    def __call__(self, offset):
        assert self.value is None
        self.value = offset
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        assert self.value is not None
        self.value = None
        return None

    def __rsub__(self, base):
        return base - self.value if self.value else base


energy_offset = _EnergyOffset()


def _get_subkwargs(func, name, mapping):
    target = mapping[func, name]
    target, override = target if isinstance(target, tuple) else (target, [])
    sub_kwargs = collect_kwarg_defaults(target, mapping)
    for x in override:
        if isinstance(x, tuple):
            key, val = x
            sub_kwargs[key] = val
        else:
            del sub_kwargs[x]
    return sub_kwargs


def collect_kwarg_defaults(func, mapping):
    kwargs = tomlkit.table()
    for p in inspect.signature(func).parameters.values():
        if p.name == 'kwargs':
            assert p.default is p.empty
            assert p.kind is inspect.Parameter.VAR_KEYWORD
            sub_kwargs = _get_subkwargs(func, 'kwargs', mapping)
            for item in sub_kwargs.value.body:
                kwargs.add(*item)
        elif p.name.endswith('_kwargs'):
            if mapping.get((func, p.name)) is True:
                kwargs[p.name] = p.default
            else:
                assert p.default is None
                assert p.kind is inspect.Parameter.KEYWORD_ONLY
                sub_kwargs = _get_subkwargs(func, p.name, mapping)
                kwargs[p.name] = sub_kwargs
        elif p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            assert p.default in (p.empty, p.default)
        else:
            assert p.kind is inspect.Parameter.KEYWORD_ONLY
            if p.default is None:
                kwargs.add(Comment(Trivia(comment=f'#: {p.name} = ...')))
            else:
                try:
                    kwargs[p.name] = p.default
                except ValueError:
                    print(func, p.name, p.kind, p.default)
                    raise
    return kwargs
