import operator
from functools import reduce

import numpy as np
import torch


def nuclear_energy(coords, charges):
    coul_IJ = charges[:, None] * charges / (coords[:, None] - coords).norm(dim=-1)
    coul_IJ[np.diag_indices(len(coords))] = 0
    return coul_IJ.sum() / 2


def nuclear_potential(r, coords, charges):
    return -(charges / (r[:, None] - coords).norm(dim=-1)).sum(-1)


def grad(r, wf, create_graph=False):
    r = r if r.requires_grad else r.detach().requires_grad_()
    psi = wf(r)
    grad_psi, = torch.autograd.grad(
        psi, r, grad_outputs=torch.ones_like(psi), create_graph=create_graph
    )
    if not create_graph:
        psi.detach_()
    return grad_psi, psi


def laplacian(r, wf, create_graph=False):
    ris = [ri.requires_grad_() for ri in r.t()]
    psi = wf(torch.stack(ris).t()).flatten()
    ones = torch.ones_like(psi)

    def d2psi_dri2_from_ri(ri):
        dpsi_dri, = torch.autograd.grad(psi, ri, grad_outputs=ones, create_graph=True)
        d2psi_dri2, = torch.autograd.grad(
            dpsi_dri,
            ri,
            grad_outputs=ones,
            retain_graph=True,
            create_graph=create_graph,
        )
        return d2psi_dri2

    lap = reduce(operator.add, map(d2psi_dri2_from_ri, ris))
    if not create_graph:
        psi.detach_()
    return lap, psi


def quantum_force(r, wf):
    grad_psi, psi = grad(r, wf)
    return grad_psi / psi[:, None], psi


def local_energy(r, wf, coords, charges, create_graph=False):
    E_nuc = nuclear_energy(coords, charges)
    V_nuc = nuclear_potential(r, coords, charges)
    lap_psi, psi = laplacian(r, wf, create_graph=create_graph)
    return -0.5 * lap_psi / psi + V_nuc + E_nuc
