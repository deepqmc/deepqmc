import torch

angstrom = 1 / 0.52917721092


def ensure_fp(tensor):
    if tensor.dtype in {torch.half, torch.float, torch.double}:
        return tensor
    return tensor.float()


class Geometry:
    def __init__(self, coords, charges):
        assert len(coords) == len(charges)
        self._coords = ensure_fp(torch.as_tensor(coords))
        self._charges = ensure_fp(torch.as_tensor(charges))

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


class Geomable:
    @property
    def geom(self):
        return Geometry(self.coords, self.charges)

    def register_geom(self, geom):
        self.register_buffer('coords', geom.coords)
        self.register_buffer('charges', geom.charges)
