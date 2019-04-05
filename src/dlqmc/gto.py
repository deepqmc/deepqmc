from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np
import pyscf.dft.numint
import torch
from scipy import special

from . import torchext

SpinTuple = namedtuple('SpinTuple', 'up down')


def eval_slater(aos, coeffs):
    if aos.shape[1] == 0:
        return 1.0
    # for molecular orbitals as linear combinations of atomic orbitals,
    # the Slater matrix can be obtained as a tensor contraction
    # (i_batch, i_elec, i_basis) * (i_basis, j_elec)
    norm = 1 / np.sqrt(special.factorial(coeffs.shape[-1]))
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


class TorchGTOSlaterWF(SlaterWF):
    def __init__(self, mf):
        SlaterWF.__init__(self, mf)
        mol = mf.mol
        self._coords = mol.atom_coords()
        self._elems = [mol.atom_symbol(i) for i in range(mol.natm)]
        self._basis = mol._basis

    def _basis_funcs(self, rs):
        for elem, coord in zip(self._elems, self._coords):
            rs_sq = ((rs - rs.new_tensor(coord)) ** 2).sum(dim=-1)
            for l, *gtos in self._basis[elem]:
                yield l, gtos, rs_sq

    def get_aos(self, rs):
        aos = []
        for l, gtos, rs_sq in self._basis_funcs(rs):
            assert l == 0
            g_exps, g_coeffs = np.array(gtos).T
            g_norms = pyscf.gto.gto_norm(l, g_exps) / np.sqrt(4 * np.pi)
            g_exps, g_norms, g_coeffs = (
                rs.new_tensor(x) for x in (g_exps, g_norms, g_coeffs)
            )
            g_contribs = g_coeffs * g_norms * torch.exp(-g_exps * rs_sq[:, None])
            aos.append(g_contribs.sum(dim=-1))
        return torch.stack(aos, dim=1)


class PyscfGTOSlaterWF(SlaterWF):
    def __init__(self, mf):
        SlaterWF.__init__(self, mf)
        self._mol = mf.mol

    def get_aos(self, rs):
        return torch.tensor(pyscf.dft.numint.eval_ao(self._mol, rs), dtype=torch.float)
