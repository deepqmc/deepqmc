from functools import partial

import torch
from torch import nn

from deepqmc.torchext import SSP, get_log_dnn
from deepqmc.utils import NULL_DEBUG

from .backflow import Backflow
from .schnet import ElectronicSchNet, SubnetFactory

__version__ = '0.2.0'


class OmniSchNet(nn.Module):
    def __init__(
        self,
        mol,
        dist_feat_dim,
        n_up,
        n_down,
        n_orbitals,
        embedding_dim=128,
        n_backflow_layers=3,
        n_jastrow_layers=3,
        with_backflow=True,
        with_jastrow=True,
        with_r_backflow=False,
        schnet_kwargs=None,
        subnet_kwargs=None,
    ):
        super().__init__()
        self.schnet = ElectronicSchNet(
            n_up,
            n_down,
            len(mol),
            dist_feat_dim=dist_feat_dim,
            embedding_dim=embedding_dim,
            subnet_metafactory=partial(SubnetFactory, **(subnet_kwargs or {})),
            **(schnet_kwargs or {}),
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

    def _get_embeddings(self, edges_elec, edges_nuc, debug):
        edges_id = id(edges_elec) + id(edges_nuc)
        if self._cache.get('dist_feats_id') != edges_id:
            self._cache['dist_feats_id'] = edges_id
            with debug.cd('schnet'):
                self._cache['embeddings'] = self.schnet(
                    edges_elec, edges_nuc, debug=debug
                )
        return self._cache['embeddings']

    def forward_jastrow(self, edges_elec, edges_nuc, debug=NULL_DEBUG):
        xs = self._get_embeddings(edges_elec, edges_nuc, debug)
        J = self.jastrow(xs.sum(dim=-2)).squeeze(dim=-1)
        return debug.result(J)

    def forward_backflow(self, mos, edges_elec, edges_nuc, debug=NULL_DEBUG):
        xs = self._get_embeddings(edges_elec, edges_nuc, debug)
        xs = debug['backflow'] = self.backflow(xs)
        return (1 + 2 * torch.tanh(xs / 4)) * mos

    def forward_r_backflow(self, rs, edges_elec, edges_nuc, debug=NULL_DEBUG):
        xs = self._get_embeddings(edges_elec, edges_nuc, debug)
        return self.r_backflow(rs, xs)

    def forward_close(self):
        self._cache.clear()
