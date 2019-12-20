from copy import deepcopy
from importlib import resources

import numpy as np
import toml
import torch
from torch import nn

__version__ = '0.1.0'
__all__ = ['Molecule']

angstrom = 1 / 0.52917721092

_SYSTEMS = toml.loads(resources.read_text('deepqmc.data', 'systems.toml'))
_SYSTEM_FACTORIES = {
    'Hn': lambda n, dist: (
        np.hstack([np.arange(n)[:, None] * dist, np.zeros((n, 2))]),
        np.ones(n),
    ),
}


def _ensure_fp(tensor):
    if tensor.dtype in {torch.half, torch.float, torch.double}:
        return tensor
    return tensor.float()


class Molecule(nn.Module):
    def __init__(self, coords, charges, charge, spin):
        assert len(coords) == len(charges)
        super().__init__()
        self.register_buffer('coords', _ensure_fp(torch.as_tensor(coords)))
        self.register_buffer('charges', _ensure_fp(torch.as_tensor(charges)))
        self.charge = charge
        self.spin = spin

    def __len__(self):
        return len(self.charges)

    def __iter__(self):
        yield from zip(self.coords, self.charges)

    def __repr__(self):
        return (
            f'Molecule(coords={self.coords}, charges={self.charges})'
            f', charge={self.charge}, spin={self.spin})'
        )

    def as_pyscf(self):
        return [(str(int(charge.numpy())), coord.numpy()) for coord, charge in self]

    @classmethod
    def from_name(cls, name, **kwargs):
        system = deepcopy(_SYSTEMS[name])
        if name in _SYSTEM_FACTORIES:
            coords, charges = _SYSTEM_FACTORIES[name](**kwargs)
        else:
            assert not kwargs
            coords = np.array(system.pop('coords'), dtype=np.float32) * angstrom
            charges = system.pop('charges')
        return cls(coords, charges, **system)
