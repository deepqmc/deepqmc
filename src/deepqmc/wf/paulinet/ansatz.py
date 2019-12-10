import torch
from torch import nn

from deepqmc.torchext import SSP, get_log_dnn
from deepqmc.utils import NULL_DEBUG

from .backflow import Backflow
from .schnet import ElectronicSchnet

__version__ = '0.2.0'


class SubnetFactory:
    def __init__(self, *, n_filter_layers, n_kernel_in_layers, n_kernel_out_layers):
        self.n_filter_layers = n_filter_layers
        self.n_kernel_in_layers = n_kernel_in_layers
        self.n_kernel_out_layers = n_kernel_out_layers

    def __call__(self, kernel_dim, embedding_dim, basis_dim):
        return (
            lambda: get_log_dnn(
                basis_dim, kernel_dim, SSP, n_layers=self.n_filter_layers
            ),
            lambda: get_log_dnn(
                embedding_dim,
                kernel_dim,
                SSP,
                last_bias=False,
                n_layers=self.n_kernel_in_layers,
            ),
            lambda: get_log_dnn(
                kernel_dim,
                embedding_dim,
                SSP,
                last_bias=False,
                n_layers=self.n_kernel_out_layers,
            ),
        )


class OmniSchnet(nn.Module):
    def __init__(
        self,
        mol,
        n_features,
        n_up,
        n_down,
        n_orbitals,
        *,
        embedding_dim,
        n_backflow_layers,
        n_jastrow_layers,
        with_backflow,
        with_jastrow,
        with_r_backflow,
        schnet_kwargs,
        schnet_subnet_kwargs,
    ):
        super().__init__()
        self.schnet = ElectronicSchnet(
            n_up,
            n_down,
            n_nuclei=len(mol),
            basis_dim=n_features,
            embedding_dim=embedding_dim,
            subnet_factories=SubnetFactory(**schnet_subnet_kwargs),
            **schnet_kwargs,
        )
        if with_jastrow:
            self.jastrow = get_log_dnn(
                embedding_dim, 1, SSP, last_bias=False, n_layers=n_jastrow_layers
            )
        else:
            self.forward_jastrow = None
        if with_backflow:
            self.backflow = get_log_dnn(
                embedding_dim,
                n_orbitals,
                SSP,
                last_bias=False,
                n_layers=n_backflow_layers,
            )
        else:
            self.forward_backflow = None
        if with_r_backflow:
            self.r_backflow = Backflow(mol, embedding_dim)
        else:
            self.forward_r_backflow = None
        self._cache = {}

    def _get_embeddings(self, edges, debug):
        if self._cache.get('edges_id') != id(edges):
            self._cache['edges_id'] = id(edges)
            with debug.cd('schnet'):
                self._cache['embeddings'] = self.schnet(edges, debug=debug)
        return self._cache['embeddings']

    def forward_jastrow(self, edges, debug=NULL_DEBUG):
        xs = self._get_embeddings(edges, debug)
        J = self.jastrow(xs.sum(dim=-2)).squeeze(dim=-1)
        return debug.result(J)

    def forward_backflow(self, mos, edges, debug=NULL_DEBUG):
        xs = self._get_embeddings(edges, debug)
        xs = debug['backflow'] = self.backflow(xs)
        return (1 + 2 * torch.tanh(xs / 4)) * mos

    def forward_r_backflow(self, rs, edges, debug=NULL_DEBUG):
        xs = self._get_embeddings(edges, debug)
        return self.r_backflow(rs, xs)

    def forward_close(self):
        self._cache.clear()
