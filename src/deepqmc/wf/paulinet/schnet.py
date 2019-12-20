from itertools import product

import numpy as np
import torch
from torch import nn

from deepqmc.torchext import SSP, get_log_dnn
from deepqmc.utils import NULL_DEBUG

from .indexing import pair_idxs, spin_pair_idxs


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
        *,
        embedding_dim,
        basis_dim,
        n_interactions=3,
        kernel_dim=64,
        subnet_factories=None,
        return_interactions=False,
        version=2,
    ):
        if not subnet_factories:

            def subnet_factories(kernel_dim, embedding_dim, basis_dim):
                return (
                    lambda: get_log_dnn(basis_dim, kernel_dim, SSP, n_layers=2),
                    lambda: nn.Linear(embedding_dim, kernel_dim, bias=False),
                    lambda: get_log_dnn(kernel_dim, embedding_dim, SSP, n_layers=2),
                )

        assert version in {1, 2}
        w_factory, h_factory, g_factory = subnet_factories(
            kernel_dim, embedding_dim, basis_dim
        )
        super().__init__()
        self.version = version
        self.n_up, self.n_down = n_up, n_down
        self.n_interactions = n_interactions
        self.return_interactions = return_interactions
        self.Y = nn.Parameter(torch.randn(n_nuclei, kernel_dim))
        self.X = nn.Parameter(torch.randn(1 if n_up == n_down else 2, embedding_dim))
        w, h, g = {}, {}, {}
        for n in range(n_interactions):
            if version == 1:
                w[f'{n}'] = w_factory()
                g[f'{n}'] = g_factory()
            elif version == 2:
                for label in [True, False, 'n']:
                    w[f'{n},{label}'] = w_factory()
                    g[f'{n},{label}'] = g_factory()
            h[f'{n}'] = h_factory()
        self.w, self.h, self.g = map(nn.ModuleDict, [w, h, g])

    def forward(self, edges_elec, edges_nuc, debug=NULL_DEBUG):
        *batch_dims, n_elec = edges_nuc.shape[:-2]
        assert edges_elec.shape[:-1] == (*batch_dims, n_elec, n_elec)
        assert n_elec == self.n_up + self.n_down
        interactions = [] if self.return_interactions else None
        x = debug[0] = torch.cat(
            [
                X.clone().expand(n, -1)
                for X, n in zip(
                    self.X, [self.n_up, self.n_down] if len(self.X) == 2 else [n_elec]
                )
            ]
        ).expand(*batch_dims, -1, -1)
        for n in range(self.n_interactions):
            h = self.h[f'{n}'](x)
            if self.version == 1:
                i, j = pair_idxs(n_elec).T
                z_elec = (
                    (self.w[f'{n}'](edges_elec[..., i, j, :]) * h[..., j, :])
                    .view(*batch_dims, n_elec, -1, h.shape[-1])
                    .sum(dim=-2)
                )
                z_nuc = (self.w[f'{n}'](edges_nuc) * self.Y[..., None, :, :]).sum(
                    dim=-2
                )
                z = self.g[f'{n}'](z_elec + z_nuc)
            elif self.version == 2:
                z_elec_uu, z_elec_ud, z_elec_du, z_elec_dd = (
                    (self.w[f'{n},{si == sj}'](edges_elec[..., i, j, :]) * h[..., j, :])
                    .view(*batch_dims, n_si, -1, h.shape[-1])
                    .sum(dim=-2)
                    for (si, sj), (i, j), n_si in zip(
                        product('ud', repeat=2),
                        spin_pair_idxs(self.n_up, self.n_down, transposed=True),
                        [self.n_up, self.n_up, self.n_down, self.n_down],
                    )
                )
                z_elec_same = torch.cat([z_elec_uu, z_elec_dd], dim=-2)
                z_elec_anti = torch.cat([z_elec_ud, z_elec_du], dim=-2)
                z_nuc = (self.w[f'{n},n'](edges_nuc) * self.Y[..., None, :, :]).sum(
                    dim=-2
                )
                z = (
                    self.g[f'{n},{True}'](z_elec_same)
                    + self.g[f'{n},{False}'](z_elec_anti)
                    + self.g[f'{n},n'](z_nuc)
                )
            if interactions is not None:
                interactions.append(z)
            x = debug[n + 1] = x + z
        return x if interactions is None else interactions
