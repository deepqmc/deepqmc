import numpy as np
import torch
from torch import nn

from ..geom import Geometry
from ..utils import NULL_DEBUG
from .anti import eval_slater
from .base import BaseWFNet, DistanceBasis, pairwise_diffs
from .cusp import CuspCorrection
from .gto import GTOBasis


class MCSCFNet(BaseWFNet):
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
        self.mo_coeff = nn.Linear(len(basis), self.activeorb, bias=False)

        if cusp_correction:
            rc = rc_scaling / self.charges.float()
            self.cusp_corr = CuspCorrection(geom.charges, self.activeorb, rc)
            self.register_buffer('basis_cusp_info', basis.get_cusp_info(rc).t())
        else:
            self.cusp_corr = None

    def init_from_pyscf(self, mf):
        mo_coeff = mf.mo_coeff.copy()
        if mf.mol.cart:
            mo_coeff *= np.sqrt(np.diag(mf.mol.intor('int1e_ovlp_cart')))[:, None]
        self.mo_coeff.weight.detach().copy_(
            torch.from_numpy(mo_coeff[:, : self.activeorb].T)
        )

    @classmethod
    def from_pyscf(cls, mf, cusp_correction=False, mo_cutoff=.0, **kwargs):
        cls.activeorb = mc.ncas
        cls.activeel  = mc.nelecas
        cls.det_list  = mc.fcisolver.large_ci(mc.ci, mc.ncas, mc.nelecas, tol=mo_cutoff, return_strs=False)
        n_up = len(cls.det_list[0][1])
        n_down =  len(cls.det_list[0][2])
        geom = Geometry(mc.mol.atom_coords().astype('float32'), mf.mol.atom_charges())
        basis = GTOBasis.from_pyscf(mc.mol)
        wf = cls(geom, n_up, n_down, basis, cusp_correction=cusp_correction)
        wf.init_from_pyscf(mc)
        return wf

    def forward(self, rs, debug=NULL_DEBUG):
        batch_dim, n_elec = rs.shape[:2]
        assert n_elec == self.n_up + self.n_down
        xs = self.orbitals(rs.flatten(end_dim=1), debug=debug)
        xs = debug['slaters'] = xs.view(batch_dim, n_elec, self.activeorb)

		re_list = list(zip(*self.det_list))
        coeff = torch.tensor(re_list[0]).cuda()
        up_index = np.array(re_list[1]).flatten()
        down_index = np.array(re_list[2]).flatten()

        det_up = eval_slater(xs[:, : self.n_up, up_index].view(-1,self.n_up,self.n_up)).view(batch_dim,-1)
        det_down = eval_slater(xs[:, self.n_up :, down_index].view(-1,self.n_down,self.n_down)).view(batch_dim,-1)

        return torch.sum(coeff * det_up * det_down,dim=-1)


    def orbitals(self, rs, debug=NULL_DEBUG):
        if self.cusp_corr:
            rs = torch.cat([self.coords, rs])  # need to know MOs at centers
        diffs_nuc = pairwise_diffs(rs, self.coords)
        aos = debug['aos'] = self.basis(diffs_nuc)
        mos = self.mo_coeff(aos)

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


    def mo_coeff_s_type_at(self, idx, xs):
        mo_coeff = self.mo_coeff.weight.t()
        mo_coeff_at = mo_coeff[self.basis.is_s_type][self.basis.s_center_idxs == idx]
        return xs @ mo_coeff_at

    def basis_cusp_info_at(self, idx):
        return self.basis_cusp_info[:, self.basis.s_center_idxs == idx]
