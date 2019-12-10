import torch
from torch import nn


class DistanceBasis(nn.Module):
    def __init__(self, basis_dim, cutoff=10.0, envelope='physnet'):
        super().__init__()
        delta = 1 / (2 * basis_dim)
        qs = torch.linspace(delta, 1 - delta, basis_dim)
        self.cutoff = cutoff
        self.envelope = envelope
        self.register_buffer('mus', cutoff * qs ** 2)
        self.register_buffer('sigmas', (1 + cutoff * qs) / 7)

    def forward(self, dists):
        if self.envelope == 'physnet':
            dists_rel = dists / self.cutoff
            envelope = torch.where(
                dists_rel > 1,
                dists.new_zeros(1),
                1 - 6 * dists_rel ** 5 + 15 * dists_rel ** 4 - 10 * dists_rel ** 3,
            )
        elif self.envelope == 'nocusp':
            envelope = dists ** 2 * torch.exp(-dists)
        else:
            assert False
        return envelope[..., None] * torch.exp(
            -((dists[..., None] - self.mus) ** 2) / self.sigmas ** 2
        )

    def extra_repr(self):
        return ', '.join(
            f'{lbl}={val!r}'
            for lbl, val in [
                ('basis_dim', len(self.mus)),
                ('cutoff', self.cutoff),
                ('envelope', self.envelope),
            ]
        )
