import torch
from torch import nn


class ElectronicAsymptotic(nn.Module):
    def __init__(self, *, cusp, alpha=1.0):
        super().__init__()
        self.cusp = cusp
        self.alpha = alpha

    def forward(self, dists):
        return torch.exp(
            -(self.cusp / (self.alpha * (1 + self.alpha * dists))).sum(dim=-1)
        )

    def extra_repr(self):
        return f'cusp={self.cusp}, alpha={self.alpha}'


# This class straightforwardly implements the cusp correction from
# http://aip.scitation.org/doi/10.1063/1.1940588. The only difference is that
# rather than deriving phi(0) by fitting to a preexisting optimal E_loc curve
# (eq. 17), we have it as a trainable parameter (self.shifts).
class CuspCorrection(nn.Module):
    def __init__(self, charges, n_orbitals, rc, eps=1e-6):
        super().__init__()
        self.register_buffer('charges', charges)
        self.shifts = nn.Parameter(torch.zeros(len(charges), n_orbitals))
        self.register_buffer('rc', rc)
        self.eps = eps

    def _fit_cusp_poly(self, phi_gto_boundary, mos0):
        has_s_part = phi_gto_boundary[0].abs() > self.eps
        charges, rc = (
            x[:, None].expand_as(has_s_part) for x in (self.charges, self.rc)
        )
        phi_gto_boundary, mos0, shifts, charges, rc = (
            x[..., has_s_part]
            for x in (phi_gto_boundary, mos0, self.shifts, charges, rc)
        )
        phi0, phi, dphi, d2phi = phi_gto_boundary
        sgn = phi0.sign()
        C = torch.where(
            (sgn == phi.sign()) & (phi0.abs() < phi.abs()),
            2 * phi0 - phi,
            2 * phi - phi0,
        )
        phi_m_C = phi - C
        X1 = torch.log(torch.abs(phi_m_C))
        X2 = dphi / phi_m_C
        X3 = d2phi / phi_m_C
        X4 = -charges * (mos0 + shifts) / (phi0 + shifts - C)
        X5 = torch.log(torch.abs(phi0 + shifts - C))
        return C, sgn, fit_cusp_poly(rc, X1, X2, X3, X4, X5), has_s_part

    def forward(self, rs_2, phi_gto_boundary, mos0):
        # TODO the indexing here is far from desirable, but I don't have time to
        # clean it up now
        C, sgn, alphas, has_s_part = self._fit_cusp_poly(phi_gto_boundary, mos0)
        rs_2_nearest, center_idx = rs_2.min(dim=-1)
        maybe_corrected = rs_2_nearest < self.rc[center_idx] ** 2
        rs_1 = rs_2_nearest[maybe_corrected].sqrt()
        corrected = maybe_corrected[:, None] & has_s_part[center_idx]
        params_idx = torch.empty_like(has_s_part, dtype=torch.long)
        params_idx[has_s_part] = torch.arange(
            has_s_part.sum(), device=params_idx.device
        )
        rs_1_idx = torch.arange(len(rs_1))[:, None].expand(-1, mos0.shape[-1])[
            corrected[maybe_corrected]
        ]
        C, sgn, *alphas = torch.stack([C, sgn, *alphas])[
            :, params_idx[center_idx][corrected]
        ]
        phi_cusped = C + sgn * eval_cusp_poly(rs_1[rs_1_idx], *alphas)
        return corrected, center_idx, phi_cusped


def fit_cusp_poly(rc, X1, X2, X3, X4, X5):
    X1_m_X5 = X1 - X5
    X2_2_m_X3 = X2 ** 2 - X3
    rc_2, rc_3, rc_4 = rc ** 2, rc ** 3, rc ** 4
    a0 = X5
    a1 = X4
    a2 = -X2_2_m_X3 / 2 - 3 * (X2 + X4) / rc + 6 * X1_m_X5 / rc_2
    a3 = X2_2_m_X3 / rc + (5 * X2 + 3 * X4) / rc_2 - 8 * X1_m_X5 / rc_3
    a4 = -X2_2_m_X3 / (2 * rc_2) - (2 * X2 + X4) / rc_3 + 3 * X1_m_X5 / rc_4
    return a0, a1, a2, a3, a4


def eval_cusp_poly(rs, a0, a1, a2, a3, a4):
    return torch.exp(a0 + a1 * rs + a2 * rs ** 2 + a3 * rs ** 3 + a4 * rs ** 4)
