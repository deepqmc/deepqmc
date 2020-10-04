import numpy as np

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
                        label, (0, *shape), maxshape=(None, *shape), dtype=dtype
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
