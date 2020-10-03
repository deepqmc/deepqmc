import torch
from torch import nn

__version__ = '0.1.0'
__all__ = ['DistanceBasis']


class DistanceBasis(nn.Module):
    r"""Expands distances in distance feature basis.

    Maps distances, *d*, to distance features, :math:`\mathbf e(d)`.

    Args:
        dist_feat_dim (int): :math:`\dim(\mathbf e)`, number of features
        cutoff (float, .a.u.): distance at which all features go to zero
        envelope (str): type of envelope for distance features

            - ``'physnet'``: taken from [UnkeJCTC19]_
            - ``'nocusp'``: used in [HermannNC20]_

    Shape:
        - Input, *d*: :math:`(*)`
        - Output, :math:`\mathbf e(d)`: :math:`(*,\dim(\mathbf e))`
    """

    def __init__(self, dist_feat_dim, cutoff=10.0, envelope='physnet', smooth=None):
        super().__init__()
        delta = 1 / (2 * dist_feat_dim)
        qs = torch.linspace(delta, 1 - delta, dist_feat_dim)
        self.cutoff = cutoff
        self.envelope = envelope
        self.register_buffer('mus', cutoff * qs ** 2)
        self.register_buffer('sigmas', (1 + cutoff * qs) / 7)
        self.smooth = smooth

    def forward(self, dists):
        if self.smooth is not None:
            dists = dists + 1 / self.smooth * torch.log(
                1 + torch.exp(-2 * self.smooth * dists)
            )
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
            raise AssertionError()
        return envelope[..., None] * torch.exp(
            -((dists[..., None] - self.mus) ** 2) / self.sigmas ** 2
        )

    def extra_repr(self):
        return ', '.join(
            f'{lbl}={val!r}'
            for lbl, val in [
                ('dist_feat_dim', len(self.mus)),
                ('cutoff', self.cutoff),
                ('envelope', self.envelope),
            ]
        )
