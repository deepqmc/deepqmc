import matplotlib.pyplot as plt
import numpy as np
import torch
from pyscf import gto

from .Examples import Potential


def plot_func(x, f, plot=plt.plot, **kwargs):
    return plot(x, f(x), **kwargs)


def get_3d_cube_mesh(bounds, npts):
    edges = [torch.linspace(*b, n) for b, n in zip(bounds, npts)]
    grids = torch.meshgrid(*edges)
    return torch.stack(grids).view(3, -1).t()


def laplacian(x, wf):

    dimension = x.shape[-1]
    batchsize = x.shape[0]

    X = []
    for i in range(dimension):
        X.append(x[:, i])
        X[i].requires_grad = True

    x = torch.cat(X, dim=0).reshape(dimension, batchsize).transpose(0, 1)
    Psi = wf(x).flatten()

    lap = 0
    for i in range(dimension):
        d = torch.autograd.grad(
            Psi,
            X[i],
            create_graph=True,
            retain_graph=True,
            grad_outputs=torch.ones(batchsize),
        )[0]
        dd = torch.autograd.grad(
            d, X[i], retain_graph=True, grad_outputs=torch.ones(batchsize)
        )[0]
        lap += dd

    return lap, Psi


def local_energy(wf, r, coords, charges):
    V = Potential(r, coords, charges)
    lap_psi, psi = laplacian(r, wf)
    return -0.5 * lap_psi / psi + V


def wf_from_mf(r, mf, i_mo):
    mol = mf.mol
    coords = torch.Tensor(mol.atom_coords())
    elems = [mol.atom_symbol(i) for i in range(mol.natm)]
    basis = mol._basis
    i_bas = 0
    psi = torch.zeros(r.shape[0])
    mo_coeff = torch.Tensor(mf.mo_coeff[:, i_mo])
    for elem, coord in zip(elems, coords):
        x_sq = ((r - coord) ** 2).sum(dim=-1)
        for l, *gtos in basis[elem]:
            assert l == 0
            g_exps, g_coeffs = torch.Tensor(gtos).t()
            g_norms = gto.gto_norm(0, g_exps) / np.sqrt(4 * np.pi)
            ao = (g_coeffs * g_norms * torch.exp(-g_exps * x_sq[:, None])).sum(dim=-1)
            psi += mo_coeff[i_bas] * ao
            i_bas += 1
    return psi
