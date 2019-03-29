import operator
from collections import namedtuple
from functools import reduce

import numpy as np
import pyscf.dft.numint
import torch
from scipy import special

from . import torchext

SpinTuple = namedtuple('SpinTuple', 'up down')


class GTOWF:
    def __init__(self, mf, occs):
        self._mo_coeffs = mf.mo_coeff[:, occs[0][0]]
        mol = mf.mol
        self._coords = mol.atom_coords()
        self._elems = [mol.atom_symbol(i) for i in range(mol.natm)]
        self._basis = mol._basis

    def __call__(self, rs):
        def basis_funcs():
            for elem, coord in zip(self._elems, self._coords):
                rs_sq = ((rs - rs.new_tensor(coord)) ** 2).sum(dim=-1)
                for l, *gtos in self._basis[elem]:
                    yield l, gtos, rs_sq

        def psiis():
            for (l, gtos, rs_sq), mo_coeff in zip(basis_funcs(), self._mo_coeffs):
                assert l == 0
                g_exps, g_coeffs = np.array(gtos).T
                g_norms = pyscf.gto.gto_norm(l, g_exps) / np.sqrt(4 * np.pi)
                g_exps, g_norms, g_coeffs = (
                    rs.new_tensor(x) for x in (g_exps, g_norms, g_coeffs)
                )
                g_contribs = g_coeffs * g_norms * torch.exp(-g_exps * rs_sq[:, None])
                yield mo_coeff * g_contribs.sum(dim=-1)

        return reduce(operator.add, psiis())


def eval_slater(aos, coeffs):
    if aos.shape[1] == 0:
        return 1.0
    return torchext.bdet((aos @ coeffs))


class PyscfGTOWF:
    def __init__(self, mf):
        self._mol = mf.mol
        self._coeffs = torch.from_numpy(mf.mo_coeff)
        self._occs = SpinTuple(
            np.isin(mf.mo_occ, [1, 2]).nonzero()[0],
            np.isin(mf.mo_occ, [2]).nonzero()[0],
        )
        assert len(self._occs[0]) + len(self._occs[1]) == mf.mo_occ.sum()

    def __call__(self, rs):
        n_samples, n_elec = rs.shape[:2]
        n_up, n_down = map(len, self._occs)
        aos = pyscf.dft.numint.eval_ao(self._mol, rs.flatten(end_dim=1))
        aos = torch.from_numpy(aos).view(n_samples, n_elec, -1)
        det_up = eval_slater(aos[:, :n_up, :], self._coeffs[:, self._occs.up])
        det_down = eval_slater(aos[:, n_up:, :], self._coeffs[:, self._occs.down])
        norm = 1 / np.sqrt(special.factorial(n_up) * special.factorial(n_down))
        return norm * det_up * det_down
