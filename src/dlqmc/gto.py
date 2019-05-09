from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
import pyscf.dft.numint
import torch
from scipy.special import factorial, factorial2
from torch import nn

from . import torchext
from .errors import DLQMCError
from .geom import Geomable, Geometry
from .nn.base import BaseWFNet
from .utils import NULL_DEBUG


def eval_slater(xs):
    if xs.shape[-1] == 0:
        return 1.0
    norm = 1 / np.sqrt(factorial(xs.shape[-1]))
    return norm * torchext.bdet(xs)


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


class SlaterWF(ABC, nn.Module, Geomable):
    def __init__(self, geom, n_up, n_down, embedding_dim):
        super().__init__()
        self.n_up, self.n_down = n_up, n_down
        self.register_geom(geom)
        self.mo = nn.Linear(embedding_dim, max(n_up, n_down), bias=False)

    def init_from_pyscf(self, mf):
        mo_coeff = mf.mo_coeff.copy()
        if mf.mol.cart:
            mo_coeff *= np.sqrt(np.diag(mf.mol.intor('int1e_ovlp_cart')))[:, None]
        self.mo.weight.detach().copy_(
            torch.from_numpy(mo_coeff[:, : max(self.n_up, self.n_down)].T)
        )

    @classmethod
    def from_pyscf(cls, mf, *args, **kwargs):
        n_up = (mf.mo_occ >= 1).sum()
        n_down = (mf.mo_occ == 2).sum()
        assert (mf.mo_occ[:n_down] == 2).all()
        assert (mf.mo_occ[n_down:n_up] == 1).all()
        assert (mf.mo_occ[n_up:] == 0).all()
        geom = Geometry(mf.mol.atom_coords().astype('float32'), mf.mol.atom_charges())
        args = args or (mf.mo_coeff.shape[0],)
        wf = cls(geom, n_up, n_down, *args, **kwargs)
        wf.init_from_pyscf(mf)
        return wf

    @abstractmethod
    def get_aos(self, rs):
        ...

    def __call__(self, rs, debug=NULL_DEBUG):
        batch_dim, n_elec = rs.shape[:2]
        xs = debug['aos'] = self.get_aos(rs.flatten(end_dim=1)).view(
            batch_dim, n_elec, -1
        )
        xs = debug['slaters'] = self.mo(xs)
        det_up = debug['det_up'] = eval_slater(xs[:, : self.n_up, : self.n_up])
        det_down = debug['det_down'] = eval_slater(xs[:, self.n_up :, : self.n_down])
        return det_up * det_down

    def orbitals(self, rs):
        xs = self.get_aos(rs)
        return self.mo(xs)

    def density(self, rs):
        xs = self.orbitals(rs)
        return sum(
            (xs[:, :n_elec] ** 2).sum(dim=-1) for n_elec in (self.n_up, self.n_down)
        )


class GTOSlaterWF(SlaterWF, BaseWFNet):
    def __init__(self, geom, n_up, n_down, basis):
        SlaterWF.__init__(self, geom, n_up, n_down, basis.dim)
        self.basis = basis

    @classmethod
    def from_pyscf(cls, mf):
        basis = GTOBasis.from_pyscf(mf.mol)
        return super().from_pyscf(mf, basis)

    def get_aos(self, rs):
        return self.basis(rs)


def TorchGTOSlaterWF(mf):
    return GTOSlaterWF.from_pyscf(mf)


def eval_ao_normed(mol, *args, **kwargs):
    aos = pyscf.dft.numint.eval_ao(mol, *args, **kwargs)
    if mol.cart:
        aos /= np.sqrt(np.diag(mol.intor('int1e_ovlp_cart')))
    return aos


class PyscfGTOSlaterWF(SlaterWF):
    def get_aos(self, rs):
        return rs.new_tensor(eval_ao_normed(self._mol, rs))

    @classmethod
    def from_pyscf(cls, mf):
        wf = super().from_pyscf(mf)
        wf._mol = mf.mol
        return wf


def electron_density_of(mf, rs):
    aos = eval_ao_normed(mf.mol, rs)
    return pyscf.dft.numint.eval_rho2(mf.mol, aos, mf.mo_coeff, mf.mo_occ, xctype='LDA')
