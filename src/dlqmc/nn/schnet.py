from itertools import product

import numpy as np
import torch
from torch import nn

from ..utils import NULL_DEBUG, nondiag
from .base import SSP, get_log_dnn


class ZeroDiagKernel(nn.Module):
    def forward(self, Ws):
        Ws = Ws.clone()
        i, j = np.diag_indices(Ws.shape[1])
        Ws[:, i, j] = 0
        return Ws


class ElectronicSchnet(nn.Module):
    def __init__(
        self,
        n_up,
        n_down,
        n_nuclei,
        n_interactions,
        basis_dim,
        kernel_dim,
        embedding_dim,
        subnet_factories=None,
        return_interactions=False,
    ):
        if not subnet_factories:

            def subnet_factories(kernel_dim, embedding_dim, basis_dim):
                return (
                    lambda: get_log_dnn(basis_dim, kernel_dim, SSP, n_layers=2),
                    lambda: nn.Linear(embedding_dim, kernel_dim, bias=False),
                    lambda: get_log_dnn(kernel_dim, embedding_dim, SSP, n_layers=2),
                )

        w_factory, h_factory, g_factory = subnet_factories(
            kernel_dim, embedding_dim, basis_dim
        )
        super().__init__()
        self.n_up = n_up
        self.n_interactions = n_interactions
        self.return_interactions = return_interactions
        self.Y = nn.Parameter(torch.randn(n_nuclei, kernel_dim))
        self.X = nn.Parameter(
            torch.cat(
                [
                    torch.randn(embedding_dim).expand(n_el, -1)
                    for n_el in ((n_up + n_down,) if n_up == n_down else (n_up, n_down))
                ]
            )
        )
        w, h, g = {}, {}, {}
        for n in range(n_interactions):
            for label in [True, False, 'n']:
                w[f'{n},{label}'] = w_factory()
                g[f'{n},{label}'] = g_factory()
            h[f'{n}'] = h_factory()
        self.w, self.h, self.g = map(nn.ModuleDict, [w, h, g])

    def forward(self, edges, debug=NULL_DEBUG):
        edges_elec, edges_nuc = edges
        *batch_dims, n_elec = edges_nuc.shape[:-2]
        assert edges_elec.shape[:-1] == (*batch_dims, n_elec, n_elec)
        interactions = [] if self.return_interactions else None
        ij = np.vstack(np.mask_indices(n_elec, nondiag)).T  # nondiagonal elements
        spin_mask = ij < self.n_up
        ijs_spin = [
            ij[spin_mask.all(axis=1)].T,
            ij[np.diff(spin_mask.astype(int))[:, 0] == -1].T,
            ij[np.diff(spin_mask.astype(int))[:, 0] == 1].T,
            ij[~spin_mask.any(axis=1)].T,
        ]  # indexes for up-up, up-down, down-up, down-down
        n_el = {'u': self.n_up, 'd': n_elec - self.n_up}
        x = debug[0] = self.X.clone().expand(*batch_dims, -1, -1)
        for n in range(self.n_interactions):
            h = self.h[f'{n}'](x)
            z_elec_uu, z_elec_ud, z_elec_du, z_elec_dd = (
                (self.w[f'{n},{si == sj}'](edges_elec[..., i, j, :]) * h[..., j, :])
                .view(*batch_dims, n_el[si], -1, h.shape[-1])
                .sum(dim=-2)
                for (si, sj), (i, j) in zip(product('ud', repeat=2), ijs_spin)
            )
            z_elec_same = torch.cat([z_elec_uu, z_elec_dd], dim=-2)
            z_elec_anti = torch.cat([z_elec_ud, z_elec_du], dim=-2)
            z_nuc = (self.w[f'{n},n'](edges_nuc) * self.Y[..., None, :, :]).sum(dim=-2)
            z = (
                self.g[f'{n},True'](z_elec_same)
                + self.g[f'{n},False'](z_elec_anti)
                + self.g[f'{n},n'](z_nuc)
            )
            if interactions is not None:
                interactions.append(z)
            x = debug[n + 1] = x + z
        return x if interactions is None else interactions
