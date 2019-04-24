import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geom import Geometry


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
        self.register_buffer('mus', cutoff * qs ** 2)
        self.register_buffer('sigmas', (1 + cutoff * qs) / 7)

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

    def extra_repr(self):
        return f'basis_dim={len(self.mus)}, cutoff={self.cutoff}'


class NuclearAsymptotic(nn.Module):
    def __init__(self, charges, ion_potential, alpha=1.0):
        super().__init__()
        self.register_buffer('charges', charges)
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

    def extra_repr(self):
        return f'alpha={self.alpha}'


def ssp(*args, **kwargs):
    return F.softplus(*args, **kwargs) - np.log(2)


class SSP(nn.Softplus):
    def forward(self, xs):
        return ssp(xs, self.beta, self.threshold)


class NetPairwiseAntisymmetry(nn.Module):
    def __init__(self, net_pair):
        super().__init__()
        self.net_pair = net_pair

    def forward(self, x_i, x_j):
        return self.net_pair(x_i, x_j) - self.net_pair(x_j, x_i)


class NetOdd(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x) - (self.net(-x))


class AntisymmetricPart(nn.Module):
    def __init__(self, net, net_pair):
        super().__init__()
        self.net_pair_anti = NetPairwiseAntisymmetry(net_pair)
        self.latentdim = list(net.parameters())[0].shape[1]
        self.net_odd = NetOdd(net)

    def forward(self, x):
        i, j = np.triu_indices(x.shape[-2], k=1)
        return self.net_odd(
            torch.prod(
                self.net_pair_anti(x[:, j].view(-1, 3), x[:, i].view(-1, 3)).view(
                    -1, len(i), self.latentdim
                ),
                dim=1,
            )
        )


class WFNet(nn.Module):
    def __init__(
        self, geom, n_electrons, ion_pot=0.5, cutoff=10.0, n_dist_feats=32, alpha=1.0
    ):
        super().__init__()
        self.register_buffer('coords', geom.coords)
        self.register_buffer('charges', geom.charges)
        self.dist_basis = DistanceBasis(n_dist_feats)
        self.nuc_asymp = NuclearAsymptotic(self.charges, ion_pot, alpha=alpha)
        n_pairs = n_electrons * len(geom) + n_electrons * (n_electrons - 1) // 2
        self.deep_lin = nn.Sequential(
            nn.Linear(n_pairs * n_dist_feats, 64),
            SSP(),
            nn.Linear(64, 64),
            SSP(),
            nn.Linear(64, 1),
        )
        self._pdist = PairwiseDistance3D()
        self._psdist = PairwiseSelfDistance3D()

    @property
    def geom(self):
        return Geometry(self.coords, self.charges)

    def _featurize(self, rs):
        dists_nuc = self._pdist(rs, self.coords[None, ...])
        dists_el = self._psdist(rs)
        dists = torch.cat([dists_nuc.flatten(start_dim=1), dists_el], dim=1)
        xs = self.dist_basis(dists).flatten(start_dim=1)
        return xs, (dists_nuc, dists_el)

    def forward(self, rs):
        xs, (dists_nuc, dists_el) = self._featurize(rs)
        ys = self.deep_lin(xs).squeeze(dim=1)
        return torch.exp(ys) * self.nuc_asymp(dists_nuc)


class WFNetAnti(nn.Module):
    def __init__(
        self,
        geom,
        n_electrons,
        net,
        net_pair,
        ion_pot=0.5,
        cutoff=10.0,
        n_dist_feats=32,
        alpha=1.0,
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
        self.antisym = AntisymmetricPart(net, net_pair)
        self._pdist = PairwiseDistance3D()
        self._psdist = PairwiseSelfDistance3D()

    def _featurize(self, rs):
        dists_nuc = self._pdist(rs, self.geom.coords[None, ...])
        dists_el = self._psdist(rs)
        dists = torch.cat([dists_nuc.flatten(start_dim=1), dists_el], dim=1)
        xs = self.dist_basis(dists)  # .flatten(start_dim=1)
        print('yo', xs.shape)
        print(rs.shape)
        return xs.flatten(start_dim=1), (dists_nuc, dists_el)

    def forward(self, rs):
        xs, (dists_nuc, dists_el) = self._featurize(rs)
        ys = self.deep_lin(xs).squeeze(dim=1)
        return torch.exp(ys) * self.nuc_asymp(dists_nuc) * self.antisym(rs)
