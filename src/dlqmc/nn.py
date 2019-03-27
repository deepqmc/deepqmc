import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WFNet(nn.Module):
    def __init__(
        self, geom, n_electrons, ion_pot=0.5, cutoff=10.0, n_dist_feats=32, alpha=1.0
    ):
        super().__init__()
        n_atoms = len(geom.charges)
        self._alpha = alpha
        self._n_elec = n_electrons
        self._cutoff = cutoff
        qs = torch.linspace(0, 1, n_dist_feats)
        self._dist_basis = nn.ParameterDict(
            {
                'mus': nn.Parameter(cutoff * qs ** 2, requires_grad=False),
                'sigmas': nn.Parameter((1 + cutoff * qs) / 7, requires_grad=False),
            }
        )
        self.geom = geom
        self.ion_pot = nn.Parameter(torch.tensor(ion_pot))
        n_pairs = n_electrons * n_atoms + n_electrons * (n_electrons - 1) // 2
        self._nn = nn.Sequential(
            nn.Linear(n_pairs * n_dist_feats, 64),
            SSP(),
            nn.Linear(64, 64),
            SSP(),
            nn.Linear(64, 1),
        )

    def _dist_envelope(self, dists):
        dists_rel = dists / self._cutoff
        return torch.where(
            dists_rel > 1,
            dists.new_zeros(1),
            1 - 6 * dists_rel ** 5 + 15 * dists_rel ** 4 - 10 * dists_rel ** 3,
        )

    def _featurize(self, rs):
        sigmas_sq = self._dist_basis.sigmas ** 2
        dists_nuc = (rs[:, :, None] - self.geom.coords).norm(dim=-1)
        i, j = np.triu_indices(self._n_elec, k=1)
        dists_el = (rs[:, :, None] - rs[:, None, :])[:, i, j].norm(dim=-1)
        dists = torch.cat([dists_nuc.flatten(start_dim=1), dists_el], dim=1)
        feats = self._dist_envelope(dists)[..., None] * torch.exp(
            -(dists[..., None] - self._dist_basis.mus) ** 2 / sigmas_sq
        )
        return feats.flatten(start_dim=1), (dists_nuc, dists_el)

    def _asymptote(self, dists_nuc, dists_el):
        tail = torch.sqrt(2 * self.ion_pot)
        asymp_nuc = (
            torch.exp(
                -(self.geom.charges * dists_nuc + tail * self._alpha * dists_nuc ** 2)
                / (1 + self._alpha * dists_nuc)
            )
            .sum(dim=-1)
            .prod(dim=-1)
        )
        return asymp_nuc

    def forward(self, rs):
        xs, dists = self._featurize(rs)
        return torch.exp(self._nn(xs).squeeze()) * self._asymptote(*dists)


def ssp(*args, **kwargs):
    return F.softplus(*args, **kwargs) - np.log(2)


class SSP(nn.Softplus):
    def forward(self, input):
        return ssp(input, self.beta, self.threshold)
