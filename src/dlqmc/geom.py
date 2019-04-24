import torch

angstrom = 1 / 0.52917721092


class Geometry:
    def __init__(self, coords, charges):
        assert len(coords) == len(charges)
        self._coords = torch.as_tensor(coords).float()
        self._charges = torch.as_tensor(charges).float()

    def __len__(self):
        return len(self._charges)

    def __iter__(self):
        yield from zip(self._coords, self._charges)

    def __repr__(self):
        return f'Geometry(coords={self._coords}, charges={self._charges})'

    @property
    def coords(self):
        return self._coords.clone()

    @property
    def charges(self):
        return self._charges.clone()

    def as_pyscf(self):
        return [(str(int(charge.numpy())), coord.numpy()) for coord, charge in self]
