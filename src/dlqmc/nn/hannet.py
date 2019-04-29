import torch
import torch.nn as nn

from ..utils import NULL_DEBUG, dctsel, triu_flat
from .anti import LaughlinAnsatz
from .base import (
    SSP,
    BaseWFNet,
    Concat,
    DistanceBasis,
    ElectronicAsymptotic,
    NuclearAsymptotic,
    get_log_dnn,
    pairwise_distance,
)
from .schnet import ElectronicSchnet


class HanNet(BaseWFNet):
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
        cusp_same=None,
        cusp_anti=None,
        **kwargs,
    ):
        super().__init__()
        self.n_up = n_up
        self.register_geom(geom)
        self.dist_basis = DistanceBasis(basis_dim, **dctsel(kwargs, 'cutoff'))
        self.asymp_nuc = NuclearAsymptotic(
            self.charges, ion_pot, **dctsel(kwargs, 'alpha')
        )
        self.asymp_same, self.asymp_anti = (
            ElectronicAsymptotic(cusp=cusp) if cusp is not None else None
            for cusp in (cusp_same, cusp_anti)
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
                # bias is subtracted by antisymmetrization anyway
                Concat(get_log_dnn(7, latent_dim, SSP, n_layers=2, last_bias=False)),
                nn.Sequential(
                    *get_log_dnn(latent_dim, 1, SSP, n_layers=2).children(),
                    nn.Sigmoid(),
                ),
            )
            if n_elec > 1
            else None
            for n_elec in (n_up, n_down)
        )

    def tracked_parameters(self):
        params = [('ion_pot', self.asymp_nuc.ion_potential)]
        for label, asymp in [
            ('cusp_same', self.asymp_same),
            ('cusp_anti', self.asymp_anti),
        ]:
            if asymp:
                params.append((label, asymp.cusp))
        return params

    def forward(self, rs, debug=NULL_DEBUG):
        dists_elec = pairwise_distance(rs, rs)
        dists_nuc = pairwise_distance(rs, self.coords[None, ...])
        dists = torch.cat([dists_elec, dists_nuc], dim=2)
        dists_basis = self.dist_basis(dists)
        with debug.cd('schnet'):
            xs = self.schnet(dists_basis, debug=debug)
        jastrow = debug['jastrow'] = self.orbital(xs).squeeze().sum(dim=-1)
        anti_up, anti_down = debug['anti_up'], debug['anti_down'] = [
            net(rs[:, idxs], dists_elec[:, idxs, idxs, None]).squeeze(dim=-1)
            if net
            else torch.tensor(1.0)
            for net, idxs in zip((self.anti_up, self.anti_down), self.spin_slices)
        ]
        asymp_nuc = debug['asymp_nuc'] = self.asymp_nuc(dists_nuc)  # TODO add electrons
        asymp_same = debug['asymp_same'] = (
            self.asymp_same(
                torch.cat(
                    [triu_flat(dists_elec[:, idxs, idxs]) for idxs in self.spin_slices],
                    dim=1,
                )
            )
            if self.asymp_same
            else 1.0
        )
        asymp_anti = debug['asymp_anti'] = (
            self.asymp_anti(
                dists_elec[:, : self.n_up, self.n_up :].flattten(start_dim=1)
            )
            if self.asymp_anti
            else 1.0
        )
        asymp = asymp_nuc * asymp_same * asymp_anti
        return anti_up * anti_down * torch.exp(jastrow) * asymp
