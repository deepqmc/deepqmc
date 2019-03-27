import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WFNet(nn.Module):
    def __init__(self, geom, ion_pot=0.5, cutoff=10.0, n_dist_feats=10, alpha=1.0):
        super().__init__()
        n_atoms = len(geom.charges)
        self._alpha = alpha
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
        self._nn = nn.Sequential(
            nn.Linear(n_atoms * n_dist_feats, 64),
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
        dists = (rs[:, None] - self.geom.coords).norm(dim=-1)
        basis = self._dist_envelope(dists)[..., None] * torch.exp(
            -(dists[..., None] - self._dist_basis.mus) ** 2 / sigmas_sq
        )
        return dists, basis.flatten(start_dim=1)

    def _asymptote(self, dists):
        tail = torch.sqrt(2 * self.ion_pot)
        return torch.exp(
            -(self.geom.charges * dists + tail * self._alpha * dists ** 2)
            / (1 + self._alpha * dists)
        ).sum(dim=-1)

    def forward(self, rs):
        dists, xs = self._featurize(rs)
        return torch.exp(self._nn(xs).squeeze()) * self._asymptote(dists)


def ssp(*args, **kwargs):
    return F.softplus(*args, **kwargs) - np.log(2)


class SSP(nn.Softplus):
    def forward(self, input):
        return ssp(input, self.beta, self.threshold)
