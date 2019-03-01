import operator
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import torch
from pyscf import gto


def plot_func(x, f, plot=plt.plot, **kwargs):
    return plot(x, f(x), **kwargs)


def get_3d_cube_mesh(bounds, npts):
    edges = [torch.linspace(*b, n) for b, n in zip(bounds, npts)]
    grids = torch.meshgrid(*edges)
    return torch.stack(grids).view(3, -1).t()


def nuclear_energy(coords, charges):
    coul_IJ = charges[:, None] * charges / (coords[:, None] - coords).norm(dim=-1)
    coul_IJ[np.diag_indices(len(coords))] = 0
    return coul_IJ.sum() / 2


def nuclear_potential(r, coords, charges):
    return -(charges / (r[:, None] - coords).norm(dim=-1)).sum(-1)


def laplacian(r, wf):
    ris = [ri.requires_grad_() for ri in r.t()]
    psi = wf(torch.stack(ris).t()).flatten()
    ones = torch.ones_like(psi)

    def d2psi_dxi2_from_xi(ri):
        dpsi_dxi, = torch.autograd.grad(psi, ri, grad_outputs=ones, create_graph=True)
        d2psi_dxi2, = torch.autograd.grad(
            dpsi_dxi, ri, grad_outputs=ones, retain_graph=True
        )
        return d2psi_dxi2

    lap = reduce(operator.add, map(d2psi_dxi2_from_xi, ris))
    return lap, psi


def local_energy(r, wf, coords, charges):
    E_nuc = nuclear_energy(coords, charges)
    V_nuc = nuclear_potential(r, coords, charges)
    lap_psi, psi = laplacian(r, wf)
    return -0.5 * lap_psi / psi + V_nuc + E_nuc


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
            g_norms = gto.gto_norm(0, g_exps) / np.sqrt(4 * np.pi)
            g_contribs = g_coeffs * g_norms * torch.exp(-g_exps * x_sq[:, None])
            yield mo_coeff * g_contribs.sum(dim=-1)

    return reduce(operator.add, psiis())
