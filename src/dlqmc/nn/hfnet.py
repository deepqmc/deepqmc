import numpy as np
import torch
from torch import nn

from ..geom import Geometry
from ..utils import NULL_DEBUG
from .anti import eval_slater
from .base import BaseWFNet, DistanceBasis, pairwise_diffs
from .cusp import CuspCorrection
from .gto import GTOBasis


class HFNet(BaseWFNet):
    def __init__(
        self,
        geom,
        n_up,
        n_down,
        basis,
        orbnet_factory=None,
        dist_basis_dim=32,
        dist_basis_cutoff=10.0,
        cusp_correction=True,
        rc_scaling=1.0,
    ):
        super().__init__()
        self.n_up, self.n_down = n_up, n_down
        self.register_geom(geom)
        self.basis = basis
        n_orb = max(n_up, n_down)
        self.mo_coeff = nn.Linear(len(basis), n_orb, bias=False)
        if orbnet_factory:
            self.orbnet = orbnet_factory(len(geom) * dist_basis_dim, n_orb)
            self.dist_basis = DistanceBasis(
                dist_basis_dim, cutoff=dist_basis_cutoff, envelope='nocusp'
            )
        else:
            self.orbnet = None
        if cusp_correction:
            rc = rc_scaling / self.charges.float()
            self.cusp_corr = CuspCorrection(geom.charges, n_orb, rc)
            self.register_buffer('basis_cusp_info', basis.get_cusp_info(rc).t())
        else:
            self.cusp_corr = None

    def init_from_pyscf(self, mf):
        mo_coeff = mf.mo_coeff.copy()
        if mf.mol.cart:
            mo_coeff *= np.sqrt(np.diag(mf.mol.intor('int1e_ovlp_cart')))[:, None]
        self.mo_coeff.weight.detach().copy_(
            torch.from_numpy(mo_coeff[:, : max(self.n_up, self.n_down)].T)
        )

    @classmethod
    def from_pyscf(cls, mf, init_mos=True, **kwargs):
        n_up = (mf.mo_occ >= 1).sum()
        n_down = (mf.mo_occ == 2).sum()
        assert (mf.mo_occ[:n_down] == 2).all()
        assert (mf.mo_occ[n_down:n_up] == 1).all()
        assert (mf.mo_occ[n_up:] == 0).all()
        geom = Geometry(mf.mol.atom_coords().astype('float32'), mf.mol.atom_charges())
        basis = GTOBasis.from_pyscf(mf.mol)
        wf = cls(geom, n_up, n_down, basis, **kwargs)
        if init_mos:
            wf.init_from_pyscf(mf)
        return wf

    def forward(self, rs, debug=NULL_DEBUG):
        batch_dim, n_elec = rs.shape[:2]
        assert n_elec == self.n_up + self.n_down
        xs = self.orbitals(rs.flatten(end_dim=1), debug=debug)
        xs = debug['slaters'] = xs.view(batch_dim, n_elec, -1)
        det_up = debug['det_up'] = eval_slater(xs[:, : self.n_up, : self.n_up])
        det_down = debug['det_down'] = eval_slater(xs[:, self.n_up :, : self.n_down])
        return det_up * det_down

    def orbitals(self, rs, debug=NULL_DEBUG):
        if self.cusp_corr:
            rs = torch.cat([self.coords, rs])  # need to know MOs at centers
        diffs_nuc = pairwise_diffs(rs, self.coords)
        aos = debug['aos'] = self.basis(diffs_nuc)
        mos = self.mo_coeff(aos)
        if self.orbnet:
            dists = diffs_nuc[:, :, 3].sqrt()
            mos = self.orbnet(mos, self.dist_basis(dists))
        if self.cusp_corr:
            n_atoms = len(self.coords)
            diffs_nuc, aos, mos, mos0 = (
                diffs_nuc[n_atoms:],
                aos[n_atoms:],
                mos[n_atoms:],
                mos[:n_atoms],
            )
            phi_gto_boundary = torch.stack(  # boundary values for s-type parts of MOs
                [
                    self.mo_coeff_s_type_at(idx, self.basis_cusp_info_at(idx))
                    for idx in range(n_atoms)
                ],
                dim=1,
            )
            corrected, center_idx, phi_cusped = self.cusp_corr(
                diffs_nuc, phi_gto_boundary, mos0
            )
            if corrected.any():
                aos = aos[corrected][:, self.basis.is_s_type]
                phi_gto = torch.empty_like(phi_cusped)
                for idx in range(n_atoms):
                    phi_gto[center_idx == idx] = self.mo_coeff_s_type_at(
                        idx, aos[center_idx == idx][:, self.basis.s_center_idxs == idx]
                    )
                mos[corrected] = mos[corrected] + phi_cusped - phi_gto
        return mos

    def density(self, rs):
        xs = self.orbitals(rs)
        return sum(
            (xs[:, :n_elec] ** 2).sum(dim=-1) for n_elec in (self.n_up, self.n_down)
        )

    def mo_coeff_s_type_at(self, idx, xs):
        mo_coeff = self.mo_coeff.weight.t()
        mo_coeff_at = mo_coeff[self.basis.is_s_type][self.basis.s_center_idxs == idx]
        return xs @ mo_coeff_at

    def basis_cusp_info_at(self, idx):
        return self.basis_cusp_info[:, self.basis.s_center_idxs == idx]
