from abc import ABC, abstractmethod
from collections import namedtuple
from functools import lru_cache

import numpy as np
import pyscf.dft.numint
import torch
from scipy.special import factorial, factorial2

from . import torchext

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


class SlaterWF(ABC):
    def __init__(self, mf):
        self._coeffs = torch.tensor(mf.mo_coeff, dtype=torch.float)
        self._occs = SpinTuple(
            np.isin(mf.mo_occ, [1, 2]).nonzero()[0],
            np.isin(mf.mo_occ, [2]).nonzero()[0],
        )
        self._mol = mf.mol
        assert len(self._occs.up) + len(self._occs.down) == mf.mo_occ.sum()

    @abstractmethod
    def get_aos(self, rs):
        ...

    def __call__(self, rs):
        n_samples, n_elec = rs.shape[:2]
        aos = self.get_aos(rs.flatten(end_dim=1)).view(n_samples, n_elec, -1)
        n_up, n_down = map(len, self._occs)
        coeffs = self._coeffs.to(aos)
        det_up = eval_slater(aos[:, :n_up, :], coeffs[:, self._occs.up])
        det_down = eval_slater(aos[:, n_up:, :], coeffs[:, self._occs.down])
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
        self._basis_info = []
        for i in range(mol.nbas):
            l = mol.bas_angular(i)
            ls = get_cartesian_angulars(l)
            coord = mol.bas_coord(i)
            zetas = mol.bas_exp(i)
            coeff_sets = mol.bas_ctr_coeff(i).T
            ovlps = np.diag(
                mol.intor('int1e_ovlp_cart', shls_slice=(i, i + 1, i, i + 1))
            )
            ovlp_sets = ovlps if len(coeff_sets) > 1 else [ovlps]
            anorms = 1 / np.sqrt(factorial2(2 * np.array(ls) - 1).prod(-1))
            rnorms = (2 * zetas / np.pi) ** (3 / 4) * (4 * zetas) ** (l / 2)
            for coeffs, ovlps in zip(coeff_sets, ovlp_sets):
                self._basis_info.append(
                    (ls, anorms * np.sqrt(ovlps), zetas, rnorms * coeffs, coord)
                )

    def get_aos(self, rs):
        aos = []
        for ls, *args in self._basis_info:
            anorms, zetas, coeffs, coord = (rs.new_tensor(arg) for arg in args)
            xs = rs - coord
            xs_sq = (xs ** 2).sum(dim=-1)
            angulars = (xs[:, None, :] ** xs.new_tensor(ls)).prod(dim=-1)
            radials = (coeffs * torch.exp(-zetas * xs_sq[:, None])).sum(dim=-1)
            aos.append(anorms * angulars * radials[:, None])
        return torch.cat(aos, dim=1)


class PyscfGTOSlaterWF(SlaterWF):
    def get_aos(self, rs):
        return rs.new_tensor(pyscf.dft.numint.eval_ao(self._mol, rs))


def electron_density_of(mf, rs):
    aos = pyscf.dft.numint.eval_ao(mf.mol, rs, deriv=0)
    return pyscf.dft.numint.eval_rho2(mf.mol, aos, mf.mo_coeff, mf.mo_occ, xctype='LDA')
