import numpy as np
import torch
import torch.nn as nn

from ..utils import NULL_DEBUG, nondiag
from .base import SSP, get_log_dnn


class ZeroDiagKernel(nn.Module):
    def forward(self, Ws):
        Ws = Ws.clone()
        i, j = np.diag_indices(Ws.shape[1])
        Ws[:, i, j] = 0
        return Ws


def get_schnet_interaction(kernel_dim, embedding_dim, basis_dim):
    modules = {
        'kernel': get_log_dnn(basis_dim, kernel_dim, SSP, n_layers=2),
        'embed_in': nn.Linear(embedding_dim, kernel_dim, bias=False),
        'embed_out': get_log_dnn(kernel_dim, embedding_dim, SSP, n_layers=2),
    }
    return nn.ModuleDict(modules)


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
    ):
        super().__init__()
        self.embedding_nuc = nn.Parameter(torch.randn(n_nuclei, kernel_dim))
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

    def forward(self, dists_basis, debug=NULL_DEBUG):
        *batch_dims, n_elec, n_all, basis_dim = dists_basis.shape
        # from a matrix with dimensions (n_elec, n_all), where n_all = n_elec +
        # n_nuclei, (c_i, c_j) select all electronic pairs excluding the
        # diagonal and all electron-nucleus pairs
        c_i, c_j, c_shape = self._conv_indexing(n_elec, n_all, batch_dims)
        dists_basis = dists_basis[..., c_i, c_j, :]
        xs = debug[0] = self.embedding_elec.clone().expand(*batch_dims, -1, -1)
        for i, interaction in enumerate(self.interactions):
            Ws = interaction.kernel(dists_basis)
            zs = interaction.embed_in(xs)
            zs = torch.cat([zs, self.embedding_nuc.expand(*batch_dims, -1, -1)], dim=1)
            zs = (Ws.view(*c_shape) * zs[:, c_j].view(*c_shape)).sum(dim=2)
            xs = debug[i + 1] = xs + interaction.embed_out(zs)
        return xs

    @staticmethod
    def _conv_indexing(n_elec, n_all, batch_dims):
        i, j = np.mask_indices(n_all, nondiag)
        n = n_elec * (n_all - 1)
        i, j = i[:n], j[:n]
        shape = (*batch_dims, n_elec, n_all - 1, -1)
        return i, j, shape
