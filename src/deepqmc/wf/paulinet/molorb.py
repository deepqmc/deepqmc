import logging

import numpy as np
import torch
from torch import nn

from deepqmc.physics import pairwise_diffs, pairwise_distance
from deepqmc.torchext import merge_tensors

from .cusp import CuspCorrection

__version__ = '0.1.0'
__all__ = ['MolecularOrbital']

log = logging.getLogger(__name__)


class MolecularOrbital(nn.Module):
    r"""Evaluates molecular orbitals from electron coordinates.

    This module combines three submodules that sequentially transform
    electron coordinates into molecular orbitals (MOs).

    1. :class:`~deepqmc.wf.paulinet.GTOBasis` transforms
       :math:`(\mathbf r-\mathbf R_I)` to basis functions
       :math:`\xi_p(\mathbf r)`
    2. :class:`torch.nn.Linear` transforms :math:`\xi_p(\mathbf r)` to
       molecular orbitals, :math:`\varphi_\mu(\mathbf r)`,

       .. math::
           \varphi_\mu(\mathbf r):=\sum_p C_{p\mu}\xi_p(\mathbf r)
    3. (optional) :class:`CuspCorrection` corrects
       :math:`\varphi_\mu(\mathbf r)` to satisfy nuclear cusp conditions

    If (3) applies, this module also determines the cusp correction cutoff radii
    as :math:`r_\text c:=q/Z`, where *q* is a global factor and *Z* is a nuclear
    charge, and if any two cutoff spheres overlap, reduces the radii accordingly.

    Args:
        mol (:class:`~deepqmc.Molecule`): target molecule
        basis (:class:`~deepqmc.wf.paulinet.GTOBasis`): basis functions
        n_orbitals (int): :math:`N_\text{orb}`, number of molecular orbitals
        cusp_correction (bool): whether the cusp correction should be applied
        rc_scaling (float): *q*, global scaling for cusp correction cutoff radii
        eps (float): numerical zero, passed to :class:`CuspCorrection`

    Shape:
        - Input1, :math:`(\mathbf r-\mathbf R_I)`: :math:`(*,M,4)`, see [dim4]_
        - Output, :math:`\varphi_\mu(\mathbf r)`: :math:`(*,N_\text{orb})`

    Attributes:
        mo_coeff: :class:`torch.nn.Linear` with no bias that represents MO coefficients
            :math:`C_{p\mu}` via its :attr:`weight` variable of shape
            :math:`(N_\text{orb},N_\text{basis})`
    """

    def __init__(
        self,
        mol,
        basis,
        n_orbitals,
        cusp_correction=True,
        rc_scaling=1.0,
        eps=1e-6,
    ):
        super().__init__()
        self.n_atoms = len(mol)
        self.n_orbitals = n_orbitals
        self.basis = basis
        self.mo_coeff = nn.Linear(len(basis), n_orbitals, bias=False)
        if cusp_correction:
            rc = rc_scaling / mol.charges.float()
            dists = pairwise_distance(mol.coords, mol.coords)
            eye = torch.eye(len(mol), out=torch.empty_like(dists))
            factors = (eye + dists / (rc + rc[:, None])).min(dim=-1).values
            if (factors < 0.99).any():
                log.warning('Reducing cusp-correction cutoffs due to overlaps')
            rc = rc * factors
            self.cusp_corr = CuspCorrection(mol.charges, n_orbitals, rc, eps=eps)
            self.register_buffer('basis_cusp_info', basis.get_cusp_info(rc).t())
        else:
            self.cusp_corr = None

    def init_from_pyscf(self, mf, freeze_mos=False):
        """Reinitialize the MO coefficient from a PySCF calculation object.

        Args:
            mf (:class:`pyscf.scf.hf.RHF` | :class:`pyscf.mcscf.mc1step.CASSCF`):
                restricted (multireference) HF calculation
            freeze_mos (bool): whether the MO coefficients should be frozen
        """
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

    def forward(self, diffs):
        # first n_atoms rows of diffs correspond to electrons on nuclei
        n_atoms = self.n_atoms
        aos = self.basis(diffs)
        mos = self.mo_coeff(aos)
        mos, mos0 = mos[n_atoms:], mos[:n_atoms]
        if self.cusp_corr:
            dists_2_nuc, aos = diffs[n_atoms:, :, 3], aos[n_atoms:]
            phi_gto_boundary = torch.stack(  # boundary values for s-type parts of MOs
                [
                    self._mo_coeff_s_type_at(idx, self._basis_cusp_info_at(idx))
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
                phi_gto[center_idx == idx] = self._mo_coeff_s_type_at(
                    idx, aos[center_idx == idx][:, self.basis.s_center_idxs == idx]
                )
            mos = merge_tensors(
                corrected,
                mos[corrected] + phi_cusped - phi_gto[corrected],
                mos[~corrected],
            )
        return mos

    def _mo_coeff_s_type_at(self, idx, xs):
        mo_coeff = self.mo_coeff.weight.t()
        mo_coeff_at = mo_coeff[self.basis.is_s_type][self.basis.s_center_idxs == idx]
        return xs @ mo_coeff_at

    def _basis_cusp_info_at(self, idx):
        return self.basis_cusp_info[:, self.basis.s_center_idxs == idx]
