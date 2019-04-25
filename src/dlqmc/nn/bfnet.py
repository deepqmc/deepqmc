import numpy as np
import torch
import torch.nn as nn

from .. import torchext
from ..geom import Geomable
from .base import SSP, DistanceBasis, NuclearAsymptotic, pairwise_distance
from .schnet import ElectronicSchnet


def get_orbnet(embedding_dim, *, n_layers):
    modules = []
    dims = [
        int(np.round(embedding_dim ** (k / n_layers)))
        for k in reversed(range(n_layers + 1))
    ]
    for k in range(n_layers):
        modules.extend([nn.Linear(dims[k], dims[k + 1]), SSP()])
    return nn.Sequential(*modules[:-1])


class BFNet(nn.Module, Geomable):
    def __init__(
        self,
        geom,
        n_up,
        n_down,
        basis_dim=32,
        kernel_dim=64,
        embedding_dim=128,
        n_interactions=3,
        n_orbital_layers=3,
        ion_pot=0.5,
        cutoff=10.0,
        alpha=1.0,
    ):
        super().__init__()
        self.n_up = n_up
        self.register_buffer('coords', geom.coords)
        self.register_buffer('charges', geom.charges)
        self.dist_basis = DistanceBasis(basis_dim)
        self.nuc_asymp = NuclearAsymptotic(self.charges, ion_pot, alpha=alpha)
        self.schnet = ElectronicSchnet(
            n_up,
            n_down,
            len(geom),
            n_interactions,
            basis_dim,
            kernel_dim,
            embedding_dim,
        )
        self.orbitals = nn.ModuleList(
            [
                get_orbnet(embedding_dim, n_layers=n_orbital_layers)
                for _ in range(n_up + n_down)
            ]
        )

    def _eval_slater(self, xs, idxs):
        phis = [orb(xs[:, idxs]) for orb in self.orbitals[idxs]]
        if not phis:
            return xs.new_ones(len(xs))
        slaters = torch.cat(phis, dim=-1)
        return torchext.bdet(slaters)

    def forward(self, rs):
        dists_elec = pairwise_distance(rs, rs)
        dists_nuc = pairwise_distance(rs, self.coords[None, ...])
        dists = torch.cat([dists_elec, dists_nuc], dim=2)
        dists_basis = self.dist_basis(dists)
        xs = self.schnet(dists_basis)
        anti_up = self._eval_slater(xs, slice(None, self.n_up))
        anti_down = self._eval_slater(xs, slice(self.n_up, None))
        return anti_up * anti_down * self.nuc_asymp(dists_nuc)
