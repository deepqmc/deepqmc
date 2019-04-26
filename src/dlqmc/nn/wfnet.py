import torch
import torch.nn as nn

from ..geom import Geomable
from ..utils import NULL_DEBUG, Debuggable, dctsel
from .anti import AntisymmetricPart
from .base import (
    SSP,
    DistanceBasis,
    NuclearAsymptotic,
    get_log_dnn,
    pairwise_distance,
    pairwise_self_distance,
)


class WFNet(nn.Module, Geomable, Debuggable):
    def __init__(
        self, geom, n_electrons, basis_dim=32, n_orbital_layers=3, ion_pot=0.5, **kwargs
    ):
        super().__init__()
        self.register_geom(geom)
        self.dist_basis = DistanceBasis(basis_dim, **dctsel(kwargs, 'cutoff'))
        self.nuc_asymp = NuclearAsymptotic(
            self.charges, ion_pot, **dctsel(kwargs, 'alpha')
        )
        n_pairs = n_electrons * len(geom) + n_electrons * (n_electrons - 1) // 2
        self.orbital = get_log_dnn(
            n_pairs * basis_dim, 1, SSP, n_layers=n_orbital_layers
        )

    def forward(self, rs, debug=NULL_DEBUG):
        dists_elec = pairwise_self_distance(rs)
        dists_nuc = pairwise_distance(rs, self.coords[None, ...])
        dists = torch.cat([dists_nuc.flatten(start_dim=1), dists_elec], dim=1)
        xs = self.dist_basis(dists).flatten(start_dim=1)
        jastrow = debug['jastrow'] = self.orbital(xs).squeeze(dim=1)
        asymp = debug['asymp'] = self.nuc_asymp(dists_nuc)
        return torch.exp(jastrow) * asymp


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

    def _featurize(self, rs):
        dists_nuc = pairwise_distance(rs, self.geom.coords[None, ...])
        dists_el = pairwise_self_distance(rs)
        dists = torch.cat([dists_nuc.flatten(start_dim=1), dists_el], dim=1)
        xs = self.dist_basis(dists)  # .flatten(start_dim=1)
        print('yo', xs.shape)
        print(rs.shape)
        return xs.flatten(start_dim=1), (dists_nuc, dists_el)

    def forward(self, rs):
        xs, (dists_nuc, dists_el) = self._featurize(rs)
        ys = self.deep_lin(xs).squeeze(dim=1)
        return torch.exp(ys) * self.nuc_asymp(dists_nuc) * self.antisym(rs)
