import operator
from functools import reduce

import numpy as np
import torch
from pyscf import gto


def wf_from_mf(r, mf, i_mo):
    mol = mf.mol
    coords = r.new_tensor(mol.atom_coords())
    elems = [mol.atom_symbol(i) for i in range(mol.natm)]
    basis = mol._basis
    mo_coeffs = r.new_tensor(mf.mo_coeff[:, i_mo])

    def basis_funcs():
        for elem, coord in zip(elems, coords):
            r_sq = ((r - coord) ** 2).sum(dim=-1)
            for l, *gtos in basis[elem]:
                yield l, gtos, r_sq

    def psiis():
        for (l, gtos, r_sq), mo_coeff in zip(basis_funcs(), mo_coeffs):
            assert l == 0
            g_exps, g_coeffs = np.array(gtos).T
            g_norms = gto.gto_norm(l, g_exps) / np.sqrt(4 * np.pi)
            g_exps, g_norms, g_coeffs = (
                r.new_tensor(x) for x in (g_exps, g_norms, g_coeffs)
            )
            g_contribs = g_coeffs * g_norms * torch.exp(-g_exps * r_sq[:, None])
            yield mo_coeff * g_contribs.sum(dim=-1)

    return reduce(operator.add, psiis())
