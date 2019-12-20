import torch
from torch import nn

from deepqmc.torchext import SSP, get_log_dnn
from deepqmc.utils import NULL_DEBUG

from .indexing import pair_idxs

__all__ = ()


def backflow_cutoff(r, L=0.5):
    r = r / L
    return torch.where(r < L, r ** 2 * (6 - 8 * r + 3 * r ** 2), r.new_ones(1))


class Backflow(nn.Module):
    def __init__(self, mol, embedding_dim, subnets_factory=None):
        if not subnets_factory:

            def subnets_factory(embedding_dim):
                return (
                    get_log_dnn(embedding_dim, 1, SSP, n_layers=3),
                    get_log_dnn(embedding_dim, len(mol), SSP, n_layers=3),
                )

        super().__init__()
        self.bf_elec, self.bf_nuc = subnets_factory(embedding_dim)

    def forward(self, rs, xs, debug=NULL_DEBUG):
        batch_dim, n_elec = rs.shape[:2]
        i, j = pair_idxs(n_elec).T
        diffs_elec = rs[..., i, :] - rs[..., j, :]
        bf_elec = (
            (
                self.bf_elec(xs[..., i, :] * xs[..., j, :]).squeeze(dim=-1)[..., None]
                * diffs_elec
            )
            .view(batch_dim, n_elec, n_elec - 1, 3)
            .sum(dim=-2)
        )
        diffs_nuc = rs[..., :, None, :] - self.mol.coords
        bf_nuc = (self.bf_nuc(xs)[..., None] * diffs_nuc).sum(dim=-2)
        cutoff = backflow_cutoff(diffs_nuc.norm(dim=-1)).prod(dim=-1)
        return rs + 1e-4 * cutoff[..., None] * (bf_elec + bf_nuc)
