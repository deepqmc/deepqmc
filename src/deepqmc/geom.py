from copy import deepcopy
from importlib import resources

import numpy as np
import toml
import torch
from torch import nn

angstrom = 1 / 0.52917721092
SYSTEMS = toml.loads(resources.read_text('deepqmc.data', 'systems.toml'))


def pairwise_distance(coords1, coords2):
    return (coords1[..., :, None, :] - coords2[..., None, :, :]).norm(dim=-1)


def pairwise_self_distance(coords):
    i, j = np.triu_indices(coords.shape[1], k=1)
    diffs = coords[..., :, None, :] - coords[..., None, :, :]
    return diffs[..., i, j, :].norm(dim=-1)


def pairwise_diffs(coords1, coords2, axes_offset=True):
    diffs = coords1[..., :, None, :] - coords2[..., None, :, :]
    if axes_offset:
        diffs = offset_from_axes(diffs)
    return torch.cat([diffs, (diffs ** 2).sum(dim=-1, keepdim=True)], dim=-1)


def diffs_to_nearest_nuc(rs, coords):
    zs = pairwise_diffs(rs, coords)
    idxs = zs[..., -1].min(dim=-1).indices
    return zs[torch.arange(len(rs)), idxs], idxs


def offset_from_axes(rs):
    eps = rs.new_tensor(100 * torch.finfo(rs.dtype).eps)
    offset = torch.where(rs < 0, -eps, eps)
    return torch.where(rs.abs() < eps, rs + offset, rs)


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


def get_H_chain(n, dist):
    return {
        'geom': Geometry(
            np.hstack([np.arange(n)[:, None] * dist, np.zeros((n, 2))]), np.ones(n)
        )
    }


def get_system(name, **kwargs):
    if name == 'Hn':
        return {**SYSTEMS[name], **get_H_chain(**kwargs)}
    assert not kwargs
    system = deepcopy(SYSTEMS[name])
    return {
        'geom': Geometry(
            np.array(system.pop('coords'), dtype=np.float32) * angstrom,
            system.pop('charges'),
        ),
        **system,
    }
