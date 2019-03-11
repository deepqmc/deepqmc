import operator
from functools import reduce

import numpy as np
import torch
from pyscf import gto


def wf_from_mf(r, mf, i_mo):
    mol = mf.mol
    coords = torch.Tensor(mol.atom_coords())
    elems = [mol.atom_symbol(i) for i in range(mol.natm)]
    basis = mol._basis
    mo_coeffs = torch.Tensor(mf.mo_coeff[:, i_mo])

    def basis_funcs():
        for elem, coord in zip(elems, coords):
            x_sq = ((r - coord) ** 2).sum(dim=-1)
            for l, *gtos in basis[elem]:
                yield l, gtos, x_sq

    def psiis():
        for (l, gtos, x_sq), mo_coeff in zip(basis_funcs(), mo_coeffs):
            assert l == 0
            g_exps, g_coeffs = torch.Tensor(gtos).t()
            g_norms = gto.gto_norm(l, g_exps) / np.sqrt(4 * np.pi)
            g_contribs = g_coeffs * g_norms * torch.exp(-g_exps * x_sq[:, None])
            yield mo_coeff * g_contribs.sum(dim=-1)

    return reduce(operator.add, psiis())
