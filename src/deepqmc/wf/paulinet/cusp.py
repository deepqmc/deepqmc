import torch
from torch import nn

__version__ = '0.1.0'
__all__ = ['ElectronicAsymptotic', 'CuspCorrection']


class ElectronicAsymptotic(nn.Module):
    r"""Jastrow factor with a correct electronic cusp.

    The Jastrow factor is calculated from distances between all pairs of
    electrons, :math:`d_{ij}`,

    .. math::
        \mathrm \gamma
        :=\sum_{ij}-\frac{c}{\alpha(1+\alpha d_{ij})}

    Args:
        cusp (float): *c*, target cusp value
        alpha (float): :math:`\alpha`, rate of decay of the cusp function to 1.

    Shape:
        - Input, :math:`d_{ij}`: :math:`(*,N_\text{pair})`
        - Output, :math:`\gamma`: :math:`(*)`
    """

    def __init__(self, *, cusp, alpha=1.0):
        super().__init__()
        self.cusp = cusp
        self.alpha = alpha

    def forward(self, dists):
        return -(self.cusp / (self.alpha * (1 + self.alpha * dists))).sum(dim=-1)

    def extra_repr(self):
        return f'cusp={self.cusp}, alpha={self.alpha}'


class CuspCorrection(nn.Module):
    r"""Corrects nuclear cusp of Gaussian-type molecular orbitals.

    The implementation closely follows [MaJCP05]_. Each orbital
    :math:`\varphi_\mu(\mathbf r)` is decomposed to its *s*-type part centered
    on the *I*-th nucleus, and the rest,

    .. math::
        \varphi_\mu(\mathbf r)=s_{\mu I}(|\mathbf r-\mathbf R_I|)
        +\phi_{\mu I}(\mathbf r)

    Within a cutoff radius, :math:`r_\text c`, the *s*-type part is replaced
    by a 4-th order polynomial that fits on four boundary values,

    .. math::
        \mathbf b_{\mu I}=\big(
            s_{\mu I}(0),
            s_{\mu I}(r_\text c),
            s_{\mu I}'(r_\text c),
            s_{\mu I}''(r_\text c)
        \big)

    The fifth coefficient of the polynomial is
    fixed by the cusp condition,

    .. math::
        \frac1{\varphi_\mu(\mathbf r)}\frac{\partial\varphi_\mu(\mathbf r)}
        {\partial|\mathbf r-\mathbf R_I|}
        \bigg|_{\mathbf r=\mathbf R_I}=-Z_I

    The value of the corrected *s*-type part at the center,
    :math:`s_{\mu I}(0)`, is further adjusted by a trainable relative shift,
    :math:`\Delta_{\mu I}`.

    Args:
        charges (:class:`~torch.Tensor`:math:`(M)`): :math:`Z_I`, nuclear charges
        n_orbitals (int): :math:`N_\text{orb}`, number of orbitals
        rc (:class:`~torch.Tensor`:math:`(M)`): :math:`r_\text c`, cutoff radii
        eps (float): :math:`\varepsilon`, numerical zero. An orbital
            is considered to have a non-zero *s*-type part if
            :math:`s_{\mu I}(0)>\varepsilon`.

    On output, the module returns the mask of corrected electron positions,
    the indexes of the nuclei that triggered the correction, and the corrected
    *s*-type part.

    Shape:
        - Input1, :math:`|\mathbf r-\mathbf R_I|^2`: :math:`(*,M)`
        - Input2, :math:`\mathbf b_{\mu I}`: :math:`(4,M,N_\text{orb})`
        - Input3, :math:`\varphi_\mu(\mathbf R_I)`: :math:`(M,N_\text{orb})`
        - Output1, corrected?: :math:`(*,N_\text{orb})`
        - Output2, which nucleus: :math:`(*)`
        - Output3, corrected :math:`s_{\mu I}(r)`: :math:`(N_\text{corr})`,
          where :math:`N_\text{corr}` is the number of nonzero elements in
          the first output (only corrected orbitals are returned in a flattened
          form)

    Attributes:
        shifts: orbital shifts :math:`\Delta_{\mu I}` of shape
            :math:`(M,N_\text{orb})`, initialized to zero
    """

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
        phi0 = phi0 * (1 + shifts)
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
        X4 = -charges * (mos0 + phi0 * shifts) / (phi0 - C)
        X5 = torch.log(torch.abs(phi0 - C))
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
