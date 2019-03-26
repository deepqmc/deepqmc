import operator
from functools import reduce

import numpy as np
import pyscf
import torch
import torch.nn as nn


class GTOWF(nn.Module):
    def __init__(self, mf, i_mo, use_pyscf=True):
        super().__init__()
        self._use_pyscf = use_pyscf
        self._mo_coeffs = mf.mo_coeff[:, i_mo]
        if use_pyscf:
            self._mol = mf.mol
        else:
            mol = mf.mol
            self._coords = mol.atom_coords()
            self._elems = [mol.atom_symbol(i) for i in range(mol.natm)]
            self._basis = mol._basis

    def forward(self, rs):
        if self._use_pyscf:
            aos = pyscf.dft.numint.eval_ao(self._mol, rs.numpy())
            return rs.new_tensor((self._mo_coeffs * aos).sum(-1))

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
