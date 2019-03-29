import torch


class Geometry:
    def __init__(self, coords, charges, dtype=torch.float):
        assert len(coords) == len(charges)
        self._coords = torch.as_tensor(coords, dtype=dtype)
        self._charges = torch.as_tensor(charges, dtype=dtype)

    def __len__(self):
        return len(self._charges)

    def __iter__(self):
        yield from zip(self._coords, self._charges)

    @property
    def coords(self):
        return self._coords.clone()

    @property
    def charges(self):
        return self._charges.clone()

    def as_pyscf(self):
        return [(str(int(charge.numpy())), coord.numpy()) for coord, charge in self]
