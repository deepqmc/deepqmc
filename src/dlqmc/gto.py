from abc import ABC, abstractmethod
from collections import namedtuple
from functools import lru_cache

import numpy as np
import pyscf.dft.numint
import torch
from scipy.special import factorial, factorial2
from torch import nn

from . import torchext
from .errors import DLQMCError
from .geom import Geomable, Geometry

SpinTuple = namedtuple('SpinTuple', 'up down')


def eval_slater(aos, coeffs):
    if aos.shape[1] == 0:
        return 1.0
    # for molecular orbitals as linear combinations of atomic orbitals,
    # the Slater matrix can be obtained as a tensor contraction
    # (i_batch, i_elec, i_basis) * (i_basis, j_elec)
    norm = 1 / np.sqrt(factorial(coeffs.shape[-1]))
    slater_matrix = aos @ coeffs
    try:
        return norm * torchext.bdet(slater_matrix)
    except torchext.LUFactError as e:
        e.info['aos'] = aos[e.info['idxs']]
        e.info['slater'] = slater_matrix[e.info['idxs']]
        raise


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


class SlaterWF(ABC, nn.Module, Geomable):
    def __init__(self, mf):
        super().__init__()
        self.register_buffer('mo_coeffs', torch.tensor(mf.mo_coeff).float())
        self.register_geom(
            Geometry(mf.mol.atom_coords().astype('float32'), mf.mol.atom_charges())
        )
        self._occs = SpinTuple(
            np.isin(mf.mo_occ, [1, 2]).nonzero()[0],
            np.isin(mf.mo_occ, [2]).nonzero()[0],
        )
        self._mol = mf.mol
        if mf.mol.cart:
            ovlps = torch.tensor(np.diag(mf.mol.intor('int1e_ovlp_cart'))).sqrt()
            self.register_buffer('ovlps', ovlps.float())
        assert len(self._occs.up) + len(self._occs.down) == mf.mo_occ.sum()

    @abstractmethod
    def get_aos(self, rs):
        ...

    def __call__(self, rs):
        n_samples, n_elec = rs.shape[:2]
        aos = self.get_aos(rs.flatten(end_dim=1)).view(n_samples, n_elec, -1)
        if self._mol.cart:
            # pyscf assumes cartesian basis is not normalized
            aos = aos * self.ovlps
        n_up, n_down = map(len, self._occs)
        det_up = eval_slater(aos[:, :n_up, :], self.mo_coeffs[:, self._occs.up])
        det_down = eval_slater(aos[:, n_up:, :], self.mo_coeffs[:, self._occs.down])
        return det_up * det_down


@lru_cache(maxsize=16)
def get_cartesian_angulars(l):
    return [
        (lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)
    ]


class TorchGTOSlaterWF(SlaterWF):
    def __init__(self, mf):
        SlaterWF.__init__(self, mf)
        mol = self._mol
        if not mol.cart:
            raise DLQMCError('TorchGTOSlaterWF supports only Cartesian basis sets')
        shells = []
        for i in range(mol.nbas):
            l = mol.bas_angular(i)
            center = torch.tensor(mol.bas_coord(i)).float()
            zetas = torch.tensor(mol.bas_exp(i)).float()
            coeff_sets = torch.tensor(mol.bas_ctr_coeff(i).T).float()
            for coeffs in coeff_sets:
                shells.append(GTOShell(center, l, coeffs, zetas))
        self.shells = nn.ModuleList(shells)

    def get_aos(self, rs):
        aos = []
        for sh in self.shells:
            xs = rs - sh.center
            xs_sq = (xs ** 2).sum(dim=-1)
            angulars = (xs[:, None, :] ** xs.new_tensor(sh.ls)).prod(dim=-1)
            radials = (sh.coeffs * torch.exp(-sh.zetas * xs_sq[:, None])).sum(dim=-1)
            aos.append(sh.anorms * angulars * radials[:, None])
        return torch.cat(aos, dim=1)


def eval_ao_normed(mol, *args, **kwargs):
    aos = pyscf.dft.numint.eval_ao(mol, *args, **kwargs)
    if mol.cart:
        aos /= np.sqrt(np.diag(mol.intor('int1e_ovlp_cart')))
    return aos


class PyscfGTOSlaterWF(SlaterWF):
    def get_aos(self, rs):
        return rs.new_tensor(eval_ao_normed(self._mol, rs))


def electron_density_of(mf, rs):
    aos = eval_ao_normed(mf.mol, rs)
    return pyscf.dft.numint.eval_rho2(mf.mol, aos, mf.mo_coeff, mf.mo_occ, xctype='LDA')
