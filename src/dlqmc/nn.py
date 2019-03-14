import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WFNet(nn.Module):
    def __init__(
        self, geom, eps=0.01, ion_pot=0.5, cutoff=10, n_dist_basis=10, alpha=1.0
    ):
        super().__init__()
        n_atoms = len(geom.charges)
        self._eps = eps
        self._cutoff = cutoff
        qs = torch.linspace(0, 1, n_dist_basis)
        self._dist_basis = {'mu': cutoff * qs ** 2, 'sigma': (1 + cutoff * qs) / 7}
        self.coords = geom.coords.clone()
        self.charges = geom.charges.clone()
        self._ion_pot = nn.Parameter(torch.tensor(ion_pot))
        self._alpha = alpha
        self._nn = nn.Sequential(
            nn.Linear(n_atoms * n_dist_basis, 64),
            SSP(),
            nn.Linear(64, 64),
            SSP(),
            nn.Linear(64, 1),
        )

    def _dist_envelope(self, dists):
        dists_rel = dists / self._cutoff
        return torch.where(
            dists_rel > 1,
            torch.zeros(1),
            1 - 6 * dists_rel ** 5 + 15 * dists_rel ** 4 - 10 * dists_rel ** 3,
        )

    def _featurize(self, rs):
        dists = (rs[:, None] - self.coords).norm(dim=-1)
        mu = self._dist_basis['mu']
        sigma_sq = self._dist_basis['sigma'] ** 2
        basis = self._dist_envelope(dists)[..., None] * torch.exp(
            -(dists[..., None] - mu) ** 2 / sigma_sq
        )
        return dists, basis.flatten(1)

    def _asymptote(self, dists):
        tail = torch.sqrt(2 * self._ion_pot)
        return torch.exp(
            -(self.charges * dists + tail * self._alpha * dists ** 2)
            / (1 + self._alpha * dists)
        ).sum(dim=-1)

    def forward(self, rs):
        dists, x = self._featurize(rs)
        return torch.exp(self._nn(x).squeeze()) * self._asymptote(dists)


def ssp(*args, **kwargs):
    return F.softplus(*args, **kwargs) - np.log(2)


class SSP(nn.Softplus):
    def forward(self, input):
        return ssp(input, self.beta, self.threshold)
