from copy import deepcopy
from importlib import resources

import numpy as np
import toml
import torch
from torch import nn

__version__ = '0.1.1'
__all__ = ['Molecule']

angstrom = 1 / 0.52917721092

_SYSTEMS = toml.loads(resources.read_text('deepqmc.data', 'systems.toml'))
_SYSTEM_FACTORIES = {
    'Hn': lambda n, dist: {
        'coords': np.hstack(
            [np.arange(n)[:, None] * dist / angstrom, np.zeros((n, 2))]
        ),
        'charges': np.ones(n),
        'charge': 0,
        'spin': n % 2,
        'unit': 'angstrom',
    },
    'H4_rect': lambda dist: {
        'coords': np.array(
            [
                [-dist / (2 * angstrom), -0.635, 0],
                [dist / (2 * angstrom), 0.635, 0],
                [-dist / (2 * angstrom), 0.635, 0],
                [dist / (2 * angstrom), -0.635, 0],
            ]
        ),
        'charges': np.ones(4),
        'charge': 0,
        'spin': 0,
        'unit': 'angstrom',
    },
}


class Molecule(nn.Module):
    """Represents a molecule.

    The array-like arguments accept anything that can be transformed to
    :class:`torch.Tensor`.

    Args:
        coords (array-like, rank-2, float, a.u.): atom coordinates as rows
        charges (array-like, rank-1, float): atom charges
        charge (int): total charge of a molecule
        spin (int): total spin multiplicity
    """

    all_names = set(_SYSTEMS.keys())

    def __init__(self, coords, charges, charge, spin, unit='bohr', data=None):
        assert len(coords) == len(charges)
        super().__init__()
        unit_multiplier = {'bohr': 1.0, 'angstrom': angstrom}[unit]
        self.register_buffer('coords', unit_multiplier * torch.as_tensor(coords))
        self.register_buffer('charges', 1.0 * torch.as_tensor(charges))
        self.charge = charge
        self.spin = spin
        self.data = data or {}

    def __len__(self):
        return len(self.charges)

    def __iter__(self):
        yield from zip(self.coords, self.charges)

    def __repr__(self):
        return (
            'Molecule(coords=\n'
            f'{self.coords.cpu().numpy()},\n'
            f'  charges={self.charges.cpu().numpy()},\n'
            f'  charge={self.charge}, spin={self.spin}\n'
            ')'
        )

    def as_pyscf(self):
        return [
            (str(int(charge.cpu().numpy())), coord.cpu().numpy())
            for coord, charge in self
        ]

    @classmethod
    def from_name(cls, name, **kwargs):
        """Create a molecule from a database of named molecules.

        The available names are in :attr:`Molecule.all_names`.
        """
        if name in _SYSTEMS:
            system = deepcopy(_SYSTEMS[name])
            system.update(kwargs)
        elif name in _SYSTEM_FACTORIES:
            system = _SYSTEM_FACTORIES[name](**kwargs)
        coords = system.pop('coords')
        return cls(coords, **system)
