import torch
from torch import nn


class CuspCorrection(nn.Module):
    def __init__(self, charges, n_orbitals, rc):
        super().__init__()
        self.register_buffer('charges', charges)
        self.shifts = nn.Parameter(torch.zeros(len(charges), n_orbitals))
        self.register_buffer('rc', rc)

    def fit_cusp_poly(self, phi_gto_boundary, mos0):
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
        X4 = -self.charges[:, None] * (mos0 + self.shifts) / (phi0 + self.shifts - C)
        X5 = torch.log(torch.abs(phi0 + self.shifts - C))
        return C, sgn, fit_cusp_poly(self.rc[:, None], X1, X2, X3, X4, X5)

    def forward(self, rs, phi_gto_boundary, mos0):
        C, sgn, alphas = self.fit_cusp_poly(phi_gto_boundary, mos0)
        rs_2_nearest, center_idx = rs[..., 3].min(dim=-1)
        mask = rs_2_nearest < self.rc[center_idx] ** 2
        center_idx = center_idx[mask]
        rs_1 = rs_2_nearest[mask].sqrt()
        C, sgn, *alphas = torch.stack([C, sgn, *alphas])[:, center_idx]
        return mask, center_idx, C + sgn * eval_cusp_poly(rs_1[:, None], *alphas)


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
