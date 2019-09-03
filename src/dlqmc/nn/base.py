import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..geom import Geomable
from ..utils import Debuggable


def pairwise_distance(coords1, coords2):
    return (coords1[..., :, None, :] - coords2[..., None, :, :]).norm(dim=-1)


def pairwise_self_distance(coords):
    i, j = np.triu_indices(coords.shape[1], k=1)
    diffs = coords[..., :, None, :] - coords[..., None, :, :]
    return diffs[..., i, j, :].norm(dim=-1)


def pairwise_diffs(coords1, coords2, axes_offset=True):
    diffs = coords1[..., :, None, :] - coords2[..., None, :, :]
    if axes_offset:
        diffs = offset_from_axes(diffs)
    return torch.cat([diffs, (diffs ** 2).sum(dim=-1, keepdim=True)], dim=-1)


def diffs_to_nearest_nuc(rs, coords):
    zs = pairwise_diffs(rs, coords)
    idxs = zs[..., -1].min(dim=-1).indices
    return zs[torch.arange(len(rs)), idxs], idxs


def offset_from_axes(rs):
    eps = rs.new_tensor(100 * torch.finfo(rs.dtype).eps)
    offset = torch.where(rs < 0, -eps, eps)
    return torch.where(rs.abs() < eps, rs + offset, rs)


class BaseWFNet(nn.Module, Geomable, Debuggable):
    def tracked_parameters(self):
        return ()

    @property
    def spin_slices(self):
        return slice(None, self.n_up), slice(self.n_up, None)


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
            -(dists[..., None] - self.mus) ** 2 / self.sigmas ** 2
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


class ElectronicAsymptotic(nn.Module):
    def __init__(self, *, cusp, alpha=1.0):
        super().__init__()
        self.cusp = cusp
        self.alpha = alpha

    def forward(self, dists):
        return torch.exp(
            -(self.cusp / (self.alpha * (1 + self.alpha * dists))).sum(dim=-1)
        )

    def extra_repr(self):
        return f'cusp={self.cusp}, alpha={self.alpha}'


def ssp(*args, **kwargs):
    return F.softplus(*args, **kwargs) - np.log(2)


class SSP(nn.Softplus):
    def forward(self, xs):
        return ssp(xs, self.beta, self.threshold)


def get_log_dnn(start_dim, end_dim, activation_factory, last_bias=True, *, n_layers):
    qs = [k / n_layers for k in range(n_layers + 1)]
    dims = [int(np.round(start_dim ** (1 - q) * end_dim ** q)) for q in qs]
    return get_custom_dnn(dims, activation_factory, last_bias=last_bias)


def get_custom_dnn(dims, activation_factory, last_bias=True):
    n_layers = len(dims) - 1
    modules = []
    for k in range(n_layers):
        bias = k + 1 < n_layers or last_bias
        modules.extend(
            [nn.Linear(dims[k], dims[k + 1], bias=bias), activation_factory()]
        )
    return nn.Sequential(*modules[:-1])


class Squeeze(nn.Module):
    def forward(self, xs):
        return xs.squeeze(dim=-1)


class Concat(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, *args):
        xs = torch.cat(args, dim=-1)
        return self.net(xs)
