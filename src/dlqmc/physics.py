import operator
from functools import reduce

import numpy as np
import torch


def nuclear_energy(geom):
    coords, charges = geom['coords'], geom['charges']
    coul_IJ = charges[:, None] * charges / (coords[:, None] - coords).norm(dim=-1)
    coul_IJ[np.diag_indices(len(coords))] = 0
    return coul_IJ.sum() / 2


def nuclear_potential(rs, geom):
    coords, charges = geom['coords'], geom['charges']
    return -(charges / (rs[:, None] - coords).norm(dim=-1)).sum(-1)


def nuclear_cusps(rs, geom):
    coords, charges = geom['coords'], geom['charges']
    return torch.exp(-charges * (rs[:, None] - coords).norm(dim=-1)).sum(dim=-1)


def grad(rs, wf, create_graph=False):
    rs = rs if rs.requires_grad else rs.detach().requires_grad_()
    psi = wf(rs)
    grad_psi, = torch.autograd.grad(
        psi, rs, grad_outputs=torch.ones_like(psi), create_graph=create_graph
    )
    if not create_graph:
        psi.detach_()
    return grad_psi, psi


def laplacian(rs, wf, create_graph=False):
    ris = [ri.requires_grad_() for ri in rs.t()]
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


def quantum_force(rs, wf):
    grad_psi, psi = grad(rs, wf)
    return grad_psi / psi[:, None], psi


def local_energy(rs, wf, geom, create_graph=False):
    E_nuc = nuclear_energy(geom)
    V_nuc = nuclear_potential(rs, geom)
    lap_psi, psi = laplacian(rs, wf, create_graph=create_graph)
    return -0.5 * lap_psi / psi + V_nuc + E_nuc, psi
