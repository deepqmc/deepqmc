import numpy as np
import torch

from .grad import grad, laplacian
from .nn.base import diffs_to_nearest_nuc
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


def quantum_force(rs, wf):
    grad_psis, psis = grad(rs, wf)
    forces = grad_psis / psis[:, None, None]
    return forces, psis


def crossover_parameter(zs, fs, charges):
    zs, zs_2 = zs[..., :3], zs[..., 3]
    zs_unit = zs / zs.norm(dim=-1)[..., None]
    fs_unit = fs / fs.norm(dim=-1)[..., None]
    Z2z2 = charges ** 2 * zs_2
    return (1 + (fs_unit * zs_unit).sum(dim=-1)) / 2 + Z2z2 / (10 * (4 + Z2z2))


def clean_force(forces, rs, geom, *, tau, return_a=False):
    zs = diffs_to_nearest_nuc(rs.flatten(end_dim=1), geom.coords).view(len(rs), -1, 4)
    a = crossover_parameter(
        zs.flatten(end_dim=1), forces.flatten(end_dim=1), geom.charges
    ).view(len(rs), -1)
    av2tau = a * (forces ** 2).sum(dim=-1) * tau
    factors = (torch.sqrt(1 + 2 * av2tau) - 1) / av2tau
    # TODO actual eps
    factors = torch.where(av2tau < 1e-7, a.new_tensor(1.0), factors)
    forces = factors[..., None] * forces
    forces_norm = forces.norm(dim=-1)
    norm_factors = torch.min(forces_norm, zs[..., -1].sqrt() / tau) / forces_norm
    forces = forces * norm_factors[..., None]
    if return_a:
        return forces, a
    return forces


def local_energy(
    rs, wf, geom=None, create_graph=False, keep_graph=None, debug=NULL_DEBUG, **kwargs
):
    geom = geom or wf.geom
    Es_nuc = debug['Es_nuc'] = nuclear_energy(geom)
    Vs_nuc = debug['Vs_nuc'] = nuclear_potential(rs, geom)
    Vs_el = debug['Vs_el'] = electronic_potential(rs)
    lap_psis, psis, *other = debug['lap_psis'], debug['psis'], *_ = laplacian(
        rs, wf, create_graph=create_graph, keep_graph=keep_graph, **kwargs
    )
    Es_loc = (
        -0.5 * lap_psis / (psis if create_graph else psis.detach())
        + Vs_nuc
        + Vs_el
        + Es_nuc
    )
    return (Es_loc, psis if keep_graph else psis.detach(), *other)
