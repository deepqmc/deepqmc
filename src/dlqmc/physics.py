import numpy as np
import torch

from .grad import grad, laplacian
from .utils import NULL_DEBUG


def nuclear_energy(geom):
    coords, charges = geom.coords, geom.charges
    coulombs = charges[:, None] * charges / (coords[:, None] - coords).norm(dim=-1)
    return coulombs.triu(1).sum()


def nuclear_potential(rs, geom):
    dists = (rs[:, :, None] - geom.coords).norm(dim=-1)
    return -(geom.charges / dists).sum(dim=(-1, -2))


def electronic_potential(rs):
    i, j = np.triu_indices(rs.shape[-2], k=1)
    dists = (rs[:, :, None] - rs[:, None, :])[:, i, j].norm(dim=-1)
    return (1 / dists).sum(dim=-1)


def quantum_force(rs, wf, *, clamp=None):
    grad_psis, psis = grad(rs, wf)
    forces = grad_psis / psis[:, None, None]
    if clamp is not None:
        clamp = rs.new_tensor(clamp)
        forces_norm = forces.norm(dim=-1)
        norm_factors = torch.min(forces_norm, clamp) / forces_norm
        forces = forces * norm_factors[..., None]
    return forces, psis


def local_energy(
    rs, wf, geom=None, create_graph=False, keep_graph=None, debug=NULL_DEBUG
):
    geom = geom or wf.geom
    Es_nuc = debug['Es_nuc'] = nuclear_energy(geom)
    Vs_nuc = debug['Vs_nuc'] = nuclear_potential(rs, geom)
    Vs_el = debug['Vs_el'] = electronic_potential(rs)
    lap_psis, psis = debug['lap_psis'], debug['psis'] = laplacian(
        rs, wf, create_graph=create_graph, keep_graph=keep_graph
    )
    return (
        -0.5 * lap_psis / (psis if create_graph else psis.detach())
        + Vs_nuc
        + Vs_el
        + Es_nuc,
        psis if keep_graph else psis.detach(),
    )
