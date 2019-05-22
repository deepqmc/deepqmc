from functools import lru_cache

import numpy as np
import torch
from scipy.special import factorial2
from torch import nn

from ..errors import DLQMCError
from ..utils import pow_int


@lru_cache(maxsize=16)
def get_cartesian_angulars(l):
    return [
        (lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)
    ]


class GTOShell(nn.Module):
    def __init__(self, l, coeffs, zetas):
        super().__init__()
        self.ls = torch.tensor(get_cartesian_angulars(l))
        anorms = 1 / np.sqrt(factorial2(2 * self.ls - 1).prod(-1))
        self.register_buffer('anorms', torch.tensor(anorms).float())
        rnorms = (2 * zetas / np.pi) ** (3 / 4) * (4 * zetas) ** (l / 2)
        self.register_buffer('coeffs', rnorms * coeffs)
        self.register_buffer('zetas', zetas)

    @property
    def l(self):
        return self.ls[0][0]

    def extra_repr(self):
        return f'l={self.l}, n_primitive={len(self.zetas)}'

    def forward(self, rs, rs_2):
        angulars = pow_int(rs[:, None, :], self.ls).prod(dim=-1)
        exps = torch.exp(-self.zetas * rs_2[:, None])
        radials = (self.coeffs * exps).sum(dim=-1)
        phis = self.anorms * angulars * radials[:, None]
        return phis


class GTOBasis(nn.Module):
    def __init__(self, centers, shells):
        super().__init__()
        self.register_buffer('centers', centers)
        self.idxs, shells = zip(*shells)
        self.shells = nn.ModuleList(shells)

    @property
    def dim(self):
        return sum(len(sh.ls) for sh in self.shells)

    @classmethod
    def from_pyscf(cls, mol):
        if not mol.cart:
            raise DLQMCError('GTOBasis supports only Cartesian basis sets')
        centers = torch.tensor(mol.atom_coords()).float()
        shells = []
        for i in range(mol.nbas):
            l = mol.bas_angular(i)
            idx = mol.bas_atom(i)
            zetas = torch.tensor(mol.bas_exp(i)).float()
            coeff_sets = torch.tensor(mol.bas_ctr_coeff(i).T).float()
            for coeffs in coeff_sets:
                shells.append((idx, GTOShell(l, coeffs, zetas))
        return cls(centers, shells)

    def forward(self, rs):
        rs = rs[:, None, :] - self.centers
        eps = rs.new_tensor(100 * torch.finfo(rs.dtype).eps)
        rs = torch.where(rs.abs() < eps, rs + eps * rs.sign(), rs)
        rs_2 = (rs ** 2).sum(dim=-1)
        shells = [
            sh(rs[:, idx], rs_2[:, idx]) for idx, sh in zip(self.idxs, self.shells)
        ]
        return torch.cat(shells, dim=-1)
