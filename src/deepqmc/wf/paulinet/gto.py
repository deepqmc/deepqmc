from functools import lru_cache

import numpy as np
import torch
from scipy.special import factorial2
from torch import nn

from deepqmc.errors import DeepQMCError
from deepqmc.torchext import fp_tensor, scatter_add

__version__ = '0.2.0'
__all__ = ['GTOBasis']


@lru_cache(maxsize=16)
def get_cartesian_angulars(l):
    return [
        (lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)
    ]


class GTOBasis(nn.Module):
    r"""Maps electron coordinates to Gaussian-type basis functions.

    The basis consists of Gaussian-type shells of various angular degree, *l*,
    centered at :math:`\mathbf R_I`, each containing :math:`(2l+1)` basis functions.

    An instance can be queried with :func:`len` to obtain the total number of
    basis functions, :math:`N_\text{basis}`.

    Args:
        centers (:class:`~torch.Tensor`:math:`(M,3)`): :math:`\mathbf R_I`, basis
            center coordinates
        shells (list): each item is a 4-tuple (:class:`int`, :class:`int`,
            :class:`~torch.Tensor`:math:`N_\text g`,
            :class:`~torch.Tensor`:math:`N_\text g`) of the center index,
            total angular momentum, Gaussian linear coefficients, and Gaussian
            exponential coefficients that represents a basis shell

    Shape:
        - Input, :math:`(\mathbf r-\mathbf R_I)`: :math:`(*,M,4)`, see [dim4]_
        - Output, :math:`\xi_p(\mathbf r)`: :math:`(*,N_\text{basis})`
    """

    def __init__(self, centers, shells):
        center_idx, ls, coeffs, zetas = zip(*shells)
        ls_cart = [torch.tensor(get_cartesian_angulars(l)) for l in ls]
        idx_ang = torch.cat([torch.full((len(l),), i) for i, l in enumerate(ls_cart)])
        idx_gto = torch.cat([torch.full((len(z),), i) for i, z in enumerate(zetas)])
        center_idx = torch.tensor(center_idx, dtype=torch.int64)
        ls = torch.tensor(ls)
        zetas, coeffs, ls_cart = map(torch.cat, (zetas, coeffs, ls_cart))
        anorms = fp_tensor(1.0 / np.sqrt(factorial2(2 * ls_cart - 1).prod(-1)))
        rnorms = (2 * zetas / np.pi) ** (3 / 4) * (4 * zetas) ** (ls[idx_gto] / 2)
        coeffs = rnorms * coeffs
        super().__init__()
        self.register_buffer('centers', centers)
        self.register_buffer('idx_gto', idx_gto)
        self.register_buffer('idx_ang', idx_ang)
        self.register_buffer('ls', ls)
        self.register_buffer('ls_cart', ls_cart)
        self.register_buffer('center_idx', center_idx)
        self.register_buffer('anorms', anorms)
        self.register_buffer('coeffs', coeffs)
        self.register_buffer('zetas', zetas)
        self.register_buffer('center_idx_s', center_idx[ls == 0])
        self.register_buffer('is_s_type', (ls_cart == 0).all(dim=-1))

    def __len__(self):
        return len(self.ls_cart)

    def get_cusp_info(self, rcs):
        rc = rcs[self.center_idx[self.idx_gto]]
        exps = torch.exp(-self.zetas * rc**2)
        phi_0 = scatter_add(self.coeffs, self.idx_gto)
        phi_rc = scatter_add(self.coeffs * exps, self.idx_gto)
        czes = self.coeffs * self.zetas * exps
        dphi_rc_dr = -2 * scatter_add(rc * czes, self.idx_gto)
        d2phi_rc_dr2 = 2 * scatter_add(
            czes * (2 * self.zetas * rc**2 - 1), self.idx_gto
        )
        x = torch.stack([phi_0, phi_rc, dphi_rc_dr, d2phi_rc_dr2], dim=-1)
        return x[self.ls == 0]

    @classmethod
    def from_pyscf(cls, mol):
        """Construct the basis from a PySCF molecule object.

        Args:
            mol (:class:`pyscf.gto.mole.Mole`): a molecule
        """
        if not mol.cart:
            raise DeepQMCError('GTOBasis supports only Cartesian basis sets')
        centers = fp_tensor(mol.atom_coords())
        shells = []
        for i in range(mol.nbas):
            l = mol.bas_angular(i)
            i_atom = mol.bas_atom(i)
            zetas = fp_tensor(mol.bas_exp(i))
            coeff_sets = fp_tensor(mol.bas_ctr_coeff(i).T)
            for coeffs in coeff_sets:
                shells.append((i_atom, l, coeffs, zetas))
        return cls(centers, shells)

    def forward(self, diffs):
        ix = self.center_idx[self.idx_ang]
        angulars = (diffs[:, ix, :3] ** self.ls_cart).prod(dim=-1)
        ix = self.center_idx[self.idx_gto]
        exps = torch.exp(-self.zetas * diffs[:, ix, -1])
        radials = scatter_add(self.coeffs * exps, self.idx_gto)
        return self.anorms * angulars * radials[:, self.idx_ang]
