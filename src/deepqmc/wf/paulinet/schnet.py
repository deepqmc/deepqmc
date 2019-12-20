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


class SubnetFactory:
    def __init__(
        self,
        dist_feat_dim,
        kernel_dim,
        embedding_dim,
        *,
        n_layers_w=2,
        n_layers_h=1,
        n_layers_g=1,
    ):
        self.dist_feat_dim = dist_feat_dim
        self.kernel_dim = kernel_dim
        self.embedding_dim = embedding_dim
        self.n_layers_w = n_layers_w
        self.n_layers_h = n_layers_h
        self.n_layers_g = n_layers_g

    def w_subnet(self):
        return get_log_dnn(
            self.dist_feat_dim, self.kernel_dim, SSP, n_layers=self.n_layers_w
        )

    def h_subnet(self):
        return get_log_dnn(
            self.embedding_dim,
            self.kernel_dim,
            SSP,
            last_bias=False,
            n_layers=self.n_layers_h,
        )

    def g_subnet(self):
        return get_log_dnn(
            self.kernel_dim,
            self.embedding_dim,
            SSP,
            last_bias=False,
            n_layers=self.n_layers_g,
        )


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
        assert version in {1, 2}
        subnet_factories = subnet_factories or SubnetFactory
        factory = subnet_factories(basis_dim, kernel_dim, embedding_dim)
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
                w[f'{n}'] = factory.w_subnet()
                g[f'{n}'] = factory.g_subnet()
            elif version == 2:
                for label in [True, False, 'n']:
                    w[f'{n},{label}'] = factory.w_subnet()
                    g[f'{n},{label}'] = factory.g_subnet()
            h[f'{n}'] = factory.h_subnet()
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
