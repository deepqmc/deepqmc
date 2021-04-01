from collections import OrderedDict
from functools import lru_cache
from itertools import combinations, permutations

# TODO remove use of numpy (torch, math)
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ()

DNN_NAMED_MODULES = True


def is_cuda(net):
    return next(net.parameters()).is_cuda


def state_dict_copy(net):
    return {name: val.cpu() for name, val in net.state_dict().items()}


def normalize_mean(x):
    return x / x.mean()


def weighted_mean_var(xs, ws):
    ws = normalize_mean(ws)
    mean = (ws * xs).mean()
    return mean, (ws * (xs - mean) ** 2).mean()


def assign_where(xs, ys, where):
    assert len(xs) == len(ys)
    for x, y in zip(xs, ys):
        x[where] = y[where]


def merge_tensors(mask, source_true, source_false):
    x = torch.empty_like(mask, dtype=source_false.dtype)
    x[mask] = source_true
    x[~mask] = source_false
    return x


def number_of_parameters(net):
    return sum(p.numel() for p in net.parameters())


def shuffle_tensor(x):
    return x[torch.randperm(len(x))]


def triu_flat(x):
    # TODO use idx_comb()
    i, j = np.triu_indices(x.shape[1], k=1)
    return x[:, i, j, ...]


def bdiag(A):
    return A.diagonal(dim1=-1, dim2=-2)


def pow_int(xs, exps):
    batch_dims = xs.shape[: -len(exps.shape)]
    zs = xs.new_zeros(*batch_dims, *exps.shape)
    xs_expanded = xs.expand_as(zs)
    for exp in exps.unique():
        mask = exps == exp
        zs[..., mask] = xs_expanded[..., mask] ** exp.item()
    return zs


def batch_eval(func, batches, *args, **kwargs):
    return torch.cat([func(batch, *args, **kwargs) for batch in batches])


def batch_eval_tuple(func, batches, *args, **kwargs):
    results = list(zip(*(func(batch, *args, **kwargs) for batch in batches)))
    return tuple(torch.cat(result) for result in results)


@lru_cache()
def idx_perm(n, r, device=torch.device('cpu')):  # noqa: B008
    idx = list(permutations(range(n), r))
    idx = torch.tensor(idx, device=device).t()
    idx = idx.view(r, *range(n, n - r, -1))
    return idx


@lru_cache()
def idx_comb(n, r, device=torch.device('cpu')):  # noqa: B008
    idx = list(combinations(range(n), r))
    idx = torch.tensor(idx, device=device).t()
    return idx


def ssp(*args, **kwargs):
    return F.softplus(*args, **kwargs) - np.log(2)


class SSP(nn.Softplus):
    def forward(self, xs):
        return ssp(xs, self.beta, self.threshold)


def get_log_dnn(start_dim, end_dim, activation_factory, last_bias=False, *, n_layers):
    qs = [k / n_layers for k in range(n_layers + 1)]
    dims = [int(np.round(start_dim ** (1 - q) * end_dim ** q)) for q in qs]
    return get_custom_dnn(dims, activation_factory, last_bias=last_bias)


def get_custom_dnn(dims, activation_factory, last_bias=False):
    n_layers = len(dims) - 1
    modules = []
    for k in range(n_layers):
        last = k + 1 == n_layers
        bias = not last or last_bias
        lin = nn.Linear(dims[k], dims[k + 1], bias=bias)
        act = activation_factory()
        modules.append((f'linear{k+1}', lin) if DNN_NAMED_MODULES else lin)
        if not last:
            modules.append((f'activ{k+1}', act) if DNN_NAMED_MODULES else act)
    if DNN_NAMED_MODULES:
        return nn.Sequential(OrderedDict(modules))
    else:
        return nn.Sequential(*modules)


def argmax_random_choice(x):
    mask = x == x.max()
    index = torch.randint(sum(mask), (1,))
    return torch.arange(len(x))[mask][index].item()
