import operator
from functools import reduce

import numpy as np
import torch


def nuclear_energy(geom):
    coords, charges = geom.coords, geom.charges
    coul_IJ = charges[:, None] * charges / (coords[:, None] - coords).norm(dim=-1)
    coul_IJ[np.diag_indices(len(coords))] = 0
    return coul_IJ.sum() / 2


def nuclear_potential(rs, geom):
    dists = (rs[:, :, None] - geom.coords).norm(dim=-1)
    return -(geom.charges / dists).sum(dim=(-1, -2))
    
def electronic_potential(rs):
	i, j = np.triu_indices(rs.shape[-2], k=1)
	dists = (rs[:,:, None] - rs[:,None, :])[:, i, j].norm(dim=-1)
	return (1 / dists).sum(dim=(-1, -2))


def grad(rs, wf, create_graph=False):
    rs = rs if rs.requires_grad else rs.detach().requires_grad_()
    psis = wf(rs)
    grad_psis, = torch.autograd.grad(
        psis, rs, grad_outputs=torch.ones_like(psis), create_graph=create_graph
    )
    if not create_graph:
        psis.detach_()
    return grad_psis, psis


def laplacian(rs, wf, create_graph=False):
    ris = [ri.requires_grad_() for ri in rs.flatten(start_dim=1).t()]
    psis = wf(torch.stack(ris, dim=1).view_as(rs))
    ones = torch.ones_like(psis)

    def d2psi_dri2_from_ri(ri):
        dpsi_dri, = torch.autograd.grad(psis, ri, grad_outputs=ones, create_graph=True)
        d2psi_dri2, = torch.autograd.grad(
            dpsi_dri,
            ri,
            grad_outputs=ones,
            retain_graph=True,
            create_graph=create_graph,
        )
        return d2psi_dri2

    lap_psis = reduce(operator.add, map(d2psi_dri2_from_ri, ris))
    if not create_graph:
        psis.detach_()
    return lap_psis, psis


def quantum_force(rs, wf):
    grad_psis, psis = grad(rs, wf)
    return grad_psis / psis[:, None, None], psis


def local_energy(rs, wf, geom, create_graph=False):
    Es_nuc = nuclear_energy(geom)
    Vs_nuc = nuclear_potential(rs, geom)
    Vs_el  = electronic_potential(rs)
    lap_psis, psis = laplacian(rs, wf, create_graph=create_graph)
    return -0.5 * lap_psis / psis + Vs_nuc + Vs_el + Es_nuc, psis
