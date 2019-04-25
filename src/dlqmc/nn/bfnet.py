import numpy as np
import torch
import torch.nn as nn

from .. import torchext
from .base import SSP, DistanceBasis, NuclearAsymptotic
from ..geom import Geomable


class ZeroDiagKernel(nn.Module):
    def forward(self, Ws):
        Ws = Ws.clone()
        i, j = np.diag_indices(Ws.shape[1])
        Ws[:, i, j] = 0
        return Ws


def get_schnet_interaction(kernel_dim, embedding_dim, basis_dim):
    modules = {
        'kernel': nn.Sequential(
            nn.Linear(basis_dim, kernel_dim),
            SSP(),
            nn.Linear(kernel_dim, kernel_dim),
            ZeroDiagKernel(),
        ),
        'embed_in': nn.Linear(embedding_dim, kernel_dim, bias=False),
        'embed_out': nn.Sequential(
            nn.Linear(kernel_dim, embedding_dim),
            SSP(),
            nn.Linear(embedding_dim, embedding_dim),
        ),
    }
    return nn.ModuleDict(modules)


def get_orbnet(embedding_dim):
    return nn.Sequential(
        nn.Linear(embedding_dim, embedding_dim // 2),
        SSP(),
        nn.Linear(embedding_dim // 2, embedding_dim // 4),
        SSP(),
        nn.Linear(embedding_dim // 4, 1),
    )


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
        ion_pot=0.5,
        cutoff=10.0,
        alpha=1.0,
    ):
        super().__init__()
        self.n_up = n_up
        self.register_buffer('coords', geom.coords)
        self.register_buffer('charges', geom.charges)
        self.dist_basis = DistanceBasis(basis_dim)
        self.asymp_nuc = NuclearAsymptotic(self.charges, ion_pot, alpha=alpha)
        self.embedding_nuc = nn.Parameter(torch.randn(len(geom), kernel_dim))
        self.embedding_elec = nn.Parameter(
            torch.cat(
                [torch.randn(embedding_dim).expand(n_el, -1) for n_el in (n_up, n_down)]
            )
        )
        self.interactions = nn.ModuleList(
            [
                get_schnet_interaction(kernel_dim, embedding_dim, basis_dim)
                for _ in range(n_interactions)
            ]
        )
        self.orbitals = nn.ModuleList(
            [get_orbnet(embedding_dim) for _ in range(n_up + n_down)]
        )

    def _eval_slater(self, xs, idxs):
        phis = [orb(xs[:, idxs]) for orb in self.orbitals[idxs]]
        if not phis:
            return xs.new_ones(len(xs))
        slaters = torch.cat(phis, dim=-1)
        return torchext.bdet(slaters)

    def forward(self, rs):
        dists_elec = (rs[:, :, None] - rs[:, None, :]).norm(dim=-1)
        dists_nuc = (rs[:, :, None] - self.coords).norm(dim=-1)
        dists = torch.cat([dists_elec, dists_nuc], dim=2)
        dists_basis = self.dist_basis(dists)
        xs = self.embedding_elec.clone().expand(len(rs), -1, -1)
        for interaction in self.interactions:
            Ws = interaction.kernel(dists_basis)
            zs = interaction.embed_in(xs)
            zs = torch.cat([zs, self.embedding_nuc.expand(len(rs), -1, -1)], dim=1)
            zs = (Ws * zs[:, None, :, :]).sum(dim=2)
            xs = xs + interaction.embed_out(zs)
        anti_up = self._eval_slater(xs, slice(None, self.n_up))
        anti_down = self._eval_slater(xs, slice(self.n_up, None))
        return anti_up * anti_down * self.asymp_nuc(dists_nuc)
