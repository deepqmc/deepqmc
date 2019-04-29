import pandas as pd
import torch
import torch.nn as nn

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

    def as_param_dict(self):
        return nn.ParameterDict(
            {
                'coords': nn.Parameter(self.coords, requires_grad=False),
                'charges': nn.Parameter(self.charges, requires_grad=False),
            }
        )


class Geomable:
    @property
    def geom(self):
        return Geometry(self.coords, self.charges)

    def register_geom(self, geom):
        self.register_buffer('coords', geom.coords)
        self.register_buffer('charges', geom.charges)


geomdb = pd.Series(
    {
        'H': Geometry([[1, 0, 0]], [1]),
        'H2+': Geometry([[-1, 0, 0], [1, 0, 0]], [1, 1]),
        'H2': Geometry([[0, 0, 0], [0.742 * angstrom, 0, 0]], [1, 1]),
        'Be': Geometry([[0, 0, 0]], [4]),
    }
)
