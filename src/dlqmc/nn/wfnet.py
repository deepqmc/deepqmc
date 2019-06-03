import torch
from torch import nn

from ..utils import NULL_DEBUG, dctsel
from .anti import AntisymmetricPart
from .base import (
    SSP,
    BaseWFNet,
    DistanceBasis,
    ElectronicAsymptotic,
    NuclearAsymptotic,
    get_log_dnn,
    pairwise_distance,
    pairwise_self_distance,
)


class WFNet(BaseWFNet):
    def __init__(
        self,
        geom,
        n_electrons,
        basis_dim=32,
        n_orbital_layers=3,
        ion_pot=0.5,
        cusp=None,
        **kwargs,
    ):
        super().__init__()
        self.register_geom(geom)
        self.dist_basis = DistanceBasis(basis_dim, **dctsel(kwargs, 'cutoff envelope'))
        self.asymp_nuc = NuclearAsymptotic(
            self.charges, ion_pot, **dctsel(kwargs, 'alpha')
        )
        self.asymp_elec = ElectronicAsymptotic(cusp=cusp) if cusp is not None else None
        n_pairs = n_electrons * len(geom) + n_electrons * (n_electrons - 1) // 2
        self.orbital = get_log_dnn(
            n_pairs * basis_dim, 1, SSP, n_layers=n_orbital_layers, last_bias=False
        )

    def tracked_parameters(self):
        params = [('ion_pot', self.asymp_nuc.ion_potential)]
        if self.asymp_elec:
            params.append(('cusp_elec', self.asymp_elec.cusp))
        return params

    def forward(self, rs, debug=NULL_DEBUG):
        dists_elec = pairwise_self_distance(rs)
        dists_nuc = pairwise_distance(rs, self.coords[None, ...])
        dists = torch.cat([dists_nuc.flatten(start_dim=1), dists_elec], dim=1)
        xs = self.dist_basis(dists).flatten(start_dim=1)
        jastrow = debug['jastrow'] = self.orbital(xs).squeeze(dim=1)
        asymp_nuc = debug['asymp_nuc'] = self.asymp_nuc(dists_nuc)
        asymp_elec = debug['asymp_elec'] = (
            self.asymp_elec(dists_elec) if self.asymp_elec else 1.0
        )
        return torch.exp(jastrow) * asymp_nuc * asymp_elec


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
        self.asymp_nuc = NuclearAsymptotic(geom.charges, ion_pot, alpha=alpha)
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
        return torch.exp(ys) * self.asymp_nuc(dists_nuc) * self.antisym(rs)
