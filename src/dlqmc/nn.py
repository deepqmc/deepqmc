import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseDistance3D(nn.Module):
    def forward(self, coords1, coords2):
        return (coords1[:, :, None] - coords2[:, None, :]).norm(dim=-1)


class PairwiseSelfDistance3D(nn.Module):
    def forward(self, coords):
        i, j = np.triu_indices(coords.shape[1], k=1)
        return (coords[:, :, None] - coords[:, None, :])[:, i, j].norm(dim=-1)


class DistanceBasis(nn.Module):
    def __init__(self, n_features, cutoff=10.0):
        super().__init__()
        qs = torch.linspace(0, 1, n_features)
        self.cutoff = cutoff
        self.mus = nn.Parameter(cutoff * qs ** 2, requires_grad=False)
        self.sigmas = nn.Parameter((1 + cutoff * qs) / 7, requires_grad=False)

    def forward(self, dists):
        dists_rel = dists / self.cutoff
        envelope = torch.where(
            dists_rel > 1,
            dists.new_zeros(1),
            1 - 6 * dists_rel ** 5 + 15 * dists_rel ** 4 - 10 * dists_rel ** 3,
        )
        return envelope[..., None] * torch.exp(
            -(dists[..., None] - self.mus) ** 2 / self.sigmas ** 2
        )


class NuclearAsymptotic(nn.Module):
    def __init__(self, charges, ion_potential, alpha=1.0):
        super().__init__()
        self.charges = nn.Parameter(charges, requires_grad=False)
        self.ion_potential = nn.Parameter(torch.as_tensor(ion_potential))
        self.alpha = alpha

    def forward(self, dists):
        decay = torch.sqrt(2 * self.ion_potential)
        return (
            torch.exp(
                -(self.charges * dists + decay * self.alpha * dists ** 2)
                / (1 + self.alpha * dists)
            )
            .sum(dim=-1)
            .prod(dim=-1)
        )


def ssp(*args, **kwargs):
    return F.softplus(*args, **kwargs) - np.log(2)


class SSP(nn.Softplus):
    def forward(self, xs):
        return ssp(xs, self.beta, self.threshold)


class WFNet(nn.Module):
    def __init__(
        self, geom, n_electrons, ion_pot=0.5, cutoff=10.0, n_dist_feats=32, alpha=1.0
    ):
        super().__init__()
        self.dist_basis = DistanceBasis(n_dist_feats)
        self.nuc_asymp = NuclearAsymptotic(geom.charges, ion_pot, alpha=alpha)
        self.geom = geom
        n_atoms = len(geom.charges)
        n_pairs = n_electrons * n_atoms + n_electrons * (n_electrons - 1) // 2
        self.deep_lin = nn.Sequential(
            nn.Linear(n_pairs * n_dist_feats, 64),
            SSP(),
            nn.Linear(64, 64),
            SSP(),
            nn.Linear(64, 1),
        )
        self._pdist = PairwiseDistance3D()
        self._psdist = PairwiseSelfDistance3D()

    def _featurize(self, rs):
        dists_nuc = self._pdist(rs, self.geom.coords[None, ...])
        dists_el = self._psdist(rs)
        dists = torch.cat([dists_nuc.flatten(start_dim=1), dists_el], dim=1)
        xs = self.dist_basis(dists).flatten(start_dim=1)
        return xs, (dists_nuc, dists_el)

    def forward(self, rs):
        xs, (dists_nuc, dists_el) = self._featurize(rs)
        ys = self.deep_lin(xs).squeeze(dim=1)
        return torch.exp(ys) * self.nuc_asymp(dists_nuc)
