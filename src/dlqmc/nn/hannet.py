import torch
import torch.nn as nn

from ..geom import Geomable
from ..utils import NULL_DEBUG, Debuggable, dctsel
from .anti import LaughlinAnsatz
from .base import (
    SSP,
    Concat,
    DistanceBasis,
    NuclearAsymptotic,
    Squeeze,
    get_log_dnn,
    pairwise_distance,
)
from .schnet import ElectronicSchnet


class HanNet(nn.Module, Geomable, Debuggable):
    def __init__(
        self,
        geom,
        n_up,
        n_down,
        basis_dim=32,
        kernel_dim=64,
        embedding_dim=128,
        latent_dim=10,
        n_interactions=3,
        n_orbital_layers=3,
        ion_pot=0.5,
        **kwargs,
    ):
        super().__init__()
        self.n_up = n_up
        self.register_geom(geom)
        self.dist_basis = DistanceBasis(basis_dim, **dctsel(kwargs, 'cutoff'))
        self.nuc_asymp = NuclearAsymptotic(
            self.charges, ion_pot, **dctsel(kwargs, 'alpha')
        )
        self.schnet = ElectronicSchnet(
            n_up,
            n_down,
            len(geom),
            n_interactions,
            basis_dim,
            kernel_dim,
            embedding_dim,
        )
        self.orbital = get_log_dnn(embedding_dim, 1, SSP, n_layers=n_orbital_layers)
        self.anti_up, self.anti_down = (
            LaughlinAnsatz(
                Concat(get_log_dnn(7, latent_dim, SSP, n_layers=2)),
                nn.Sequential(
                    *get_log_dnn(latent_dim, 1, SSP, n_layers=2).children(),
                    Squeeze(),
                    nn.Sigmoid(),
                ),
            )
            if n_elec > 1
            else None
            for n_elec in (n_up, n_down)
        )

    def forward(self, rs, debug=NULL_DEBUG):
        dists_elec = pairwise_distance(rs, rs)
        dists_nuc = pairwise_distance(rs, self.coords[None, ...])
        dists = torch.cat([dists_elec, dists_nuc], dim=2)
        dists_basis = self.dist_basis(dists)
        with debug.cd('schnet'):
            xs = self.schnet(dists_basis, debug=debug)
        jastrow = debug['jastrow'] = self.orbital(xs).squeeze().sum(dim=-1)
        anti_up, anti_down = debug['anti_up'], debug['anti_down'] = [
            net(rs[:, idxs], dists_elec[:, idxs, idxs, None])
            if net
            else torch.tensor(1.0)
            for net, idxs in [
                (self.anti_up, slice(None, self.n_up)),
                (self.anti_down, slice(self.n_up, None)),
            ]
        ]
        asymp = debug['asymp'] = self.nuc_asymp(dists_nuc)  # TODO add electrons
        return anti_up * anti_down * torch.exp(jastrow) * asymp
