import numpy as np
import torch

from .grad import grad, laplacian
from .utils import NULL_DEBUG

__all__ = ()


def pairwise_distance(coords1, coords2):
    return (coords1[..., :, None, :] - coords2[..., None, :, :]).norm(dim=-1)


def pairwise_self_distance(coords):
    i, j = np.triu_indices(coords.shape[1], k=1)
    diffs = coords[..., :, None, :] - coords[..., None, :, :]
    return diffs[..., i, j, :].norm(dim=-1)


def pairwise_diffs(coords1, coords2, axes_offset=True):
    diffs = coords1[..., :, None, :] - coords2[..., None, :, :]
    if axes_offset:
        diffs = offset_from_axes(diffs)
    return torch.cat([diffs, (diffs ** 2).sum(dim=-1, keepdim=True)], dim=-1)


def diffs_to_nearest_nuc(rs, coords):
    zs = pairwise_diffs(rs, coords)
    idxs = zs[..., -1].min(dim=-1).indices
    return zs[torch.arange(len(rs)), idxs], idxs


def offset_from_axes(rs):
    eps = rs.new_tensor(100 * torch.finfo(rs.dtype).eps)
    offset = torch.where(rs < 0, -eps, eps)
    return torch.where(rs.abs() < eps, rs + offset, rs)


def nuclear_energy(mol):
    coords, charges = mol.coords, mol.charges
    coulombs = charges[:, None] * charges / (coords[:, None] - coords).norm(dim=-1)
    return coulombs.triu(1).sum()


def nuclear_potential(rs, mol):
    dists = (rs[:, :, None] - mol.coords).norm(dim=-1)
    return -(mol.charges / dists).sum(dim=(-1, -2))


def electronic_potential(rs):
    i, j = np.triu_indices(rs.shape[-2], k=1)
    dists = (rs[:, :, None] - rs[:, None, :])[:, i, j].norm(dim=-1)
    return (1 / dists).sum(dim=-1)


def quantum_force(rs, wf):
    grad_psis, psis = grad(rs, wf)
    forces = grad_psis / psis[:, None, None]
    return forces, psis


# The two following functions implement the techniques from
# https://aip.scitation.org/doi/10.1063/1.465195. 'crossover_parameter'
# implements directly eq. 36 from there. 'clean_force' implements eq. 35 and a
# simplified version of the treatment of the force close to nuclei. Rather than
# projecting the force to cylindrical coordinates, we clamp the norm of the
# force such that f*tau is never longer than the distance to the closest
# nucleus.


def crossover_parameter(zs, fs, charges):
    zs, zs_2 = zs[..., :3], zs[..., 3]
    zs_unit = zs / zs.norm(dim=-1)[..., None]
    fs_unit = fs / fs.norm(dim=-1)[..., None]
    Z2z2 = charges ** 2 * zs_2
    return (1 + (fs_unit * zs_unit).sum(dim=-1)) / 2 + Z2z2 / (10 * (4 + Z2z2))


def clean_force(forces, rs, mol, *, tau, return_a=False):
    zs, idxs = diffs_to_nearest_nuc(rs.flatten(end_dim=1), mol.coords)
    zs = zs.view(len(rs), -1, 4)
    a = crossover_parameter(
        zs.flatten(end_dim=1), forces.flatten(end_dim=1), mol.charges[idxs]
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
    rs, wf, mol=None, create_graph=False, keep_graph=None, debug=NULL_DEBUG, **kwargs
):
    mol = mol or wf.mol
    Es_nuc = debug['Es_nuc'] = nuclear_energy(mol)
    Vs_nuc = debug['Vs_nuc'] = nuclear_potential(rs, mol)
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
