from functools import lru_cache

import numpy as np
import torch
from scipy.special import factorial2
from torch import nn

from ..errors import DLQMCError


@lru_cache(maxsize=16)
def get_cartesian_angulars(l):
    return [
        (lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)
    ]


class GTOShell(nn.Module):
    def __init__(self, center, l, coeffs, zetas):
        super().__init__()
        self.ls = get_cartesian_angulars(l)
        self.register_buffer('center', center)
        anorms = 1 / np.sqrt(factorial2(2 * np.array(self.ls) - 1).prod(-1))
        self.register_buffer('anorms', torch.tensor(anorms).float())
        rnorms = (2 * zetas / np.pi) ** (3 / 4) * (4 * zetas) ** (l / 2)
        self.register_buffer('coeffs', rnorms * coeffs)
        self.register_buffer('zetas', zetas)

    def extra_repr(self):
        return f'l={self.ls[0][0]}, n_primitive={len(self.zetas)}'

    def forward(self, rs):
        xs = rs - self.center
        eps = rs.new_tensor(torch.finfo(rs.dtype).eps)
        xs = torch.where(xs.abs() > eps, xs, eps)
        xs_sq = (xs ** 2).sum(dim=-1)
        angulars = (xs[:, None, :] ** xs.new_tensor(self.ls)).prod(dim=-1)
        radials = (self.coeffs * torch.exp(-self.zetas * xs_sq[:, None])).sum(dim=-1)
        return self.anorms * angulars * radials[:, None]


class GTOBasis(nn.Module):
    def __init__(self, shells):
        super().__init__()
        self.shells = nn.ModuleList(shells)

    @property
    def dim(self):
        return sum(len(sh.ls) for sh in self.shells)

    @classmethod
    def from_pyscf(cls, mol):
        if not mol.cart:
            raise DLQMCError('GTOBasis supports only Cartesian basis sets')
        shells = []
        for i in range(mol.nbas):
            l = mol.bas_angular(i)
            center = torch.tensor(mol.bas_coord(i)).float()
            zetas = torch.tensor(mol.bas_exp(i)).float()
            coeff_sets = torch.tensor(mol.bas_ctr_coeff(i).T).float()
            for coeffs in coeff_sets:
                shells.append(GTOShell(center, l, coeffs, zetas))
        return cls(shells)

    def forward(self, rs):
        return torch.cat([sh(rs) for sh in self.shells], dim=-1)
