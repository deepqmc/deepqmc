import numpy as np
import pandas as pd
import torch
from torch import nn

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

    def cpu(self):
        return Geometry(self._coords.cpu(), self._charges.cpu())

    def cuda(self):
        return Geometry(self._coords.cuda(), self._charges.cuda())

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
        label: Geometry(np.array(coords, dtype=np.float32) * angstrom, charges)
        for label, coords, charges in [
            ('H', [[1, 0, 0]], [1]),
            ('H2+', [[-1 / angstrom, 0, 0], [1 / angstrom, 0, 0]], [1, 1]),
            ('H2', [[0, 0, 0], [0.742, 0, 0]], [1, 1]),
            ('Be', [[0, 0, 0]], [4]),
            ('B', [[0, 0, 0]], [5]),
            ('LiH', [[0, 0, 0], [1.595, 0, 0]], [3, 1]),
            (
                'H2O',
                [
                    [0.0000, 0.0000, 0.1173],
                    [0.0000, 0.7572, -0.4692],
                    [0.0000, -0.7572, -0.4692],
                ],
                [8, 1, 1],
            ),
        ]
    }
)
