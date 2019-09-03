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
        for i in range(n_interactions):
            w[f'{i}'] = w_factory()
            g[f'{i}'] = g_factory()
            h[f'{i}'] = h_factory()
        self.w, self.h, self.g = map(nn.ModuleDict, [w, h, g])

    def forward(self, edges, debug=NULL_DEBUG):
        edges_elec, edges_nuc = edges
        *batch_dims, n_elec, n_atoms, basis_dim = edges_nuc.shape
        assert edges_elec.shape == (*batch_dims, n_elec, n_elec, basis_dim)
        i, j = np.mask_indices(n_elec, nondiag)  # nondiagonal elements
        edges_elec = edges_elec[..., i, j, :]
        x = debug[0] = self.X.clone().expand(*batch_dims, -1, -1)
        interactions = [] if self.return_interactions else None
        for i in range(self.n_interactions):
            z_elec = (
                (self.w[f'{i}'](edges_elec) * self.h[f'{i}'](x)[:, j])
                .view(*batch_dims, n_elec, n_elec - 1, -1)
                .sum(dim=-2)
            )
            z_nuc = (self.w[f'{i}'](edges_nuc) * self.Y[..., None, :, :]).sum(dim=-2)
            z = self.g[f'{i}'](z_elec + z_nuc)
            if interactions is not None:
                interactions.append(z)
            x = debug[i + 1] = x + z
        return x if interactions is None else interactions
