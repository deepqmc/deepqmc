import numpy as np
import torch
from torch import nn

from ..utils import NULL_DEBUG
from .base import SSP, get_log_dnn,conv_indexing,DistanceBasis,pairwise_distance


class Backflow(nn.Module):
      
    def __init__(
        self,
        n_up,
        n_down,
        n_interactions,
        basis_dim,
        n_layers,
    ):
        super().__init__()
        self.n_up = n_up
        self.n_down = n_down
        self.dist_basis = DistanceBasis(basis_dim)
        self.interactions = nn.ModuleList(
            [
                get_log_dnn(basis_dim, 1, SSP, n_layers=n_layers, last_bias=False)
                for _ in range(n_interactions)
            ])
            
    def forward(self,rs, debug=NULL_DEBUG):
        
        t = []
        for xs in [rs[:,:self.n_up].clone(),rs[:,self.n_up:].clone()]:
            for i, interaction in enumerate(self.interactions):
                dists_basis = self.dist_basis(pairwise_distance(xs,xs))
                *batch_dims, n_elec, n_elec, basis_dim = dists_basis.shape
                c_i, c_j, c_shape = conv_indexing(n_elec, n_elec, batch_dims)
                dists_basis = dists_basis[..., c_i, c_j, :]
                Ws = interaction(dists_basis)
                ys = (Ws.view(*c_shape) * (xs[:, c_j].view(*c_shape)-xs[:,:,None,:])).sum(dim=2)
                xs = debug[i] = xs + ys
            t.append(xs)
        return torch.cat(t,dim=1)
    


