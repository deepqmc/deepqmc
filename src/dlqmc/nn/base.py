import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_distance(coords1, coords2):
    return (coords1[:, :, None] - coords2[:, None, :]).norm(dim=-1)


def pairwise_self_distance(coords):
    i, j = np.triu_indices(coords.shape[1], k=1)
    return (coords[:, :, None] - coords[:, None, :])[:, i, j].norm(dim=-1)


class PairwiseDistance3D(nn.Module):
    def forward(self, coords1, coords2):
        return pairwise_distance(coords1, coords2)


class PairwiseSelfDistance3D(nn.Module):
    def forward(self, coords):
        return pairwise_self_distance(coords)


class DistanceBasis(nn.Module):
    def __init__(self, basis_dim, cutoff=10.0):
        super().__init__()
        qs = torch.linspace(0, 1, basis_dim)
        self.cutoff = cutoff
        self.register_buffer('mus', cutoff * qs ** 2)
        self.register_buffer('sigmas', (1 + cutoff * qs) / 7)

    def forward(self, dists):
        dists_rel = dists / self.cutoff
        envelope = torch.where(
            dists_rel > 1,
            dists.new_zeros(1),
            1 - 6 * dists_rel ** 5 + 15 * dists_rel ** 4 - 10 * dists_rel ** 3,
        )
        return envelope[..., None] * torch.exp(
            -(dists[..., None] - self.mus) ** 2 / self.sigmas ** 2
        )

    def extra_repr(self):
        return f'basis_dim={len(self.mus)}, cutoff={self.cutoff}'


class NuclearAsymptotic(nn.Module):
    def __init__(self, charges, ion_potential, alpha=1.0):
        super().__init__()
        self.register_buffer('charges', charges)
        self.ion_potential = nn.Parameter(torch.as_tensor(ion_potential))
        self.alpha = alpha

    def forward(self, dists):
        decay = torch.sqrt(2 * self.ion_potential)
        return (
            torch.exp(
                -(self.charges * dists + decay * self.alpha * dists ** 2)
                / (1 + self.alpha * dists)
            )
            .sum(dim=-1)
            .prod(dim=-1)
        )

    def extra_repr(self):
        return f'alpha={self.alpha}'


def ssp(*args, **kwargs):
    return F.softplus(*args, **kwargs) - np.log(2)


class SSP(nn.Softplus):
    def forward(self, xs):
        return ssp(xs, self.beta, self.threshold)
