import torch
import torch.nn as nn

from ..geom import Geomable
from ..utils import NULL_DEBUG, Debuggable
from .anti import AntisymmetricPart
from .base import (
    SSP,
    DistanceBasis,
    NuclearAsymptotic,
    PairwiseDistance3D,
    PairwiseSelfDistance3D,
)


class WFNet(nn.Module, Geomable, Debuggable):
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

    def forward(self, rs, debug=NULL_DEBUG):
        dists_nuc = self._pdist(rs, self.coords[None, ...])
        dists_el = self._psdist(rs)
        dists = torch.cat([dists_nuc.flatten(start_dim=1), dists_el], dim=1)
        xs = self.dist_basis(dists).flatten(start_dim=1)
        ys = debug['ys'] = self.deep_lin(xs).squeeze(dim=1)
        asymp = debug['asymp'] = self.nuc_asymp(dists_nuc)
        return torch.exp(ys) * asymp


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
