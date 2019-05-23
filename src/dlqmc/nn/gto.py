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

    def __len__(self):
        return len(self.ls)

    @property
    def l(self):
        return self.ls[0][0]

    def extra_repr(self):
        return f'l={self.l}, n_primitive={len(self.zetas)}'

    def get_cusp_info(self, rc):
        assert self.l == 0
        exps = torch.exp(-self.zetas * rc ** 2)
        phi_0 = self.coeffs.sum()
        phi_rc = (self.coeffs * exps).sum()
        czes = self.coeffs * self.zetas * exps
        dphi_rc_dr = -2 * rc * czes.sum()
        d2phi_rc_dr2 = 2 * (czes * (2 * self.zetas * rc ** 2 - 1)).sum()
        return torch.stack([phi_0, phi_rc, dphi_rc_dr, d2phi_rc_dr2])

    def forward(self, rs):
        rs, rs_2 = rs[..., :3], rs[..., 3]
        angulars = pow_int(rs[:, None, :], self.ls).prod(dim=-1)
        exps = torch.exp(-self.zetas * rs_2[:, None])
        radials = (self.coeffs * exps).sum(dim=-1)
        phis = self.anorms * angulars * radials[:, None]
        return phis


class GTOBasis(nn.Module):
    def __init__(self, centers, shells):
        super().__init__()
        self.register_buffer('centers', centers)
        self.center_idxs, shells = zip(*shells)
        self.shells = nn.ModuleList(shells)
        self.s_center_idxs = torch.tensor(
            [idx for idx, sh in self.items() if sh.l == 0]
        )
        self.is_s_type = torch.cat(
            [
                (torch.ones if sh.l == 0 else torch.zeros)(len(sh), dtype=torch.uint8)
                for sh in self.shells
            ]
        )

    def __len__(self):
        return sum(map(len, self.shells))

    def items(self):
        return zip(self.center_idxs, self.shells)

    def get_cusp_info(self, rc):
        return torch.stack([sh.get_cusp_info(rc) for sh in self.shells if sh.l == 0])

    @classmethod
    def from_pyscf(cls, mol):
        if not mol.cart:
            raise DLQMCError('GTOBasis supports only Cartesian basis sets')
        centers = torch.tensor(mol.atom_coords()).float()
        shells = []
        for i in range(mol.nbas):
            l = mol.bas_angular(i)
            i_atom = mol.bas_atom(i)
            zetas = torch.tensor(mol.bas_exp(i)).float()
            coeff_sets = torch.tensor(mol.bas_ctr_coeff(i).T).float()
            for coeffs in coeff_sets:
                shells.append((i_atom, GTOShell(l, coeffs, zetas)))
        return cls(centers, shells)

    def forward(self, diffs):
        shells = [sh(diffs[:, idx]) for idx, sh in self.items()]
        return torch.cat(shells, dim=-1)
