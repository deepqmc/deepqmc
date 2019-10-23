from logging import warning

import numpy as np
import torch
from torch import nn

from ..torchext import merge_tensors
from ..utils import NULL_DEBUG
from .base import pairwise_diffs, pairwise_distance
from .cusp import CuspCorrection


class MolecularOrbital(nn.Module):
    def __init__(
        self,
        geom,
        basis,
        n_orbitals,
        net_factory=None,
        edge_dim=None,
        cusp_correction=True,
        rc_scaling=1.0,
        eps=1e-6,
    ):
        super().__init__()
        self.n_atoms = len(geom)
        self.n_orbitals = n_orbitals
        self.basis = basis
        self.mo_coeff = nn.Linear(len(basis), n_orbitals, bias=False)
        self.net = net_factory(len(geom), edge_dim, n_orbitals) if net_factory else None
        if cusp_correction:
            rc = rc_scaling / geom.charges.float()
            dists = pairwise_distance(geom.coords, geom.coords)
            eye = torch.eye(len(geom), out=torch.empty_like(dists))
            factors = (eye + dists / (rc + rc[:, None])).min(dim=-1).values
            if (factors < 0.99).any():
                warning('Reducing cusp-correction cutoffs due to overlaps')
            rc = rc * factors
            self.cusp_corr = CuspCorrection(geom.charges, n_orbitals, rc, eps=eps)
            self.register_buffer('basis_cusp_info', basis.get_cusp_info(rc).t())
        else:
            self.cusp_corr = None

    def init_from_pyscf(self, mf, freeze_mos=False):
        mo_coeff = mf.mo_coeff.copy()
        if mf.mol.cart:
            mo_coeff *= np.sqrt(np.diag(mf.mol.intor('int1e_ovlp_cart')))[:, None]
        self.mo_coeff.weight.detach().copy_(
            torch.from_numpy(mo_coeff[:, : self.n_orbitals].T)
        )
        if freeze_mos:
            self.mo_coeff.weight.requires_grad_(False)

    def forward_from_rs(self, rs, coords):
        diffs_nuc = pairwise_diffs(torch.cat([coords, rs]), coords)
        return self(diffs_nuc)

    def forward(self, diffs, edges=None, debug=NULL_DEBUG):
        # first n_atoms rows of diffs and edges correspond to electrons on
        # nuclei
        n_atoms = self.n_atoms
        aos = debug['aos'] = self.basis(diffs)
        mos = self.mo_coeff(aos)
        if self.net:
            mos = self.net(mos, edges)
        mos, mos0 = mos[n_atoms:], mos[:n_atoms]
        if self.cusp_corr:
            dists_2_nuc, aos = diffs[n_atoms:, :, 3], aos[n_atoms:]
            phi_gto_boundary = torch.stack(  # boundary values for s-type parts of MOs
                [
                    self.mo_coeff_s_type_at(idx, self.basis_cusp_info_at(idx))
                    for idx in range(n_atoms)
                ],
                dim=1,
            )
            corrected, center_idx, phi_cusped = self.cusp_corr(
                dists_2_nuc, phi_gto_boundary, mos0
            )
            aos = aos[:, self.basis.is_s_type]
            phi_gto = torch.empty_like(mos)
            for idx in range(n_atoms):
                if not (center_idx == idx).any():
                    continue
                phi_gto[center_idx == idx] = self.mo_coeff_s_type_at(
                    idx, aos[center_idx == idx][:, self.basis.s_center_idxs == idx]
                )
            mos = merge_tensors(
                corrected,
                mos[corrected] + phi_cusped - phi_gto[corrected],
                mos[~corrected],
            )
        return debug.result(mos)

    def mo_coeff_s_type_at(self, idx, xs):
        mo_coeff = self.mo_coeff.weight.t()
        mo_coeff_at = mo_coeff[self.basis.is_s_type][self.basis.s_center_idxs == idx]
        return xs @ mo_coeff_at

    def basis_cusp_info_at(self, idx):
        return self.basis_cusp_info[:, self.basis.s_center_idxs == idx]
