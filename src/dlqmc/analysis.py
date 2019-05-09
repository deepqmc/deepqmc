import numpy as np
import torch

from .nn.base import pairwise_distance, pairwise_self_distance
from .stats import GaussianKDEstimator
from .utils import shuffle_tensor


def blocking(xs, max_B=None):
    N = xs.shape[1]
    x_sigma = xs.std()
    max_B = max_B or int(np.log2(N))
    sigmas_B = []
    for log_B in range(0, max_B):
        B = 2 ** log_B
        sigmas_B.append(
            xs[:, -(N // B * B) :]
            .view(xs.shape[0], -1, B)
            .mean(dim=-1)
            .std(dim=-1)
            .mean()
            * np.sqrt(B)
            / x_sigma
        )
    return torch.tensor(sigmas_B)


def autocorr_coeff(ks, xs):
    x_mean = xs.mean()
    x_var = xs.var()
    Cs = []
    for k in ks:
        end = -k or xs.shape[1]
        x_autocov = ((xs[:, :end] - x_mean) * (xs[:, k:] - x_mean)).mean()
        Cs.append(x_autocov / x_var)
    return torch.tensor(Cs)


def pair_correlations_from_samples(rs, n_up, bw=0.1):
    R_uu = pairwise_self_distance(rs[:, :n_up]).flatten()
    R_dd = pairwise_self_distance(rs[:, n_up:]).flatten()
    R_ud = pairwise_distance(rs[:, :n_up], rs[:, n_up:]).flatten()
    rs_decorr = shuffle_tensor(rs.view(-1, 3)).view(-1, 2, 3)
    R_decorr = pairwise_self_distance(rs_decorr)[:, 0]
    return {
        'uu': GaussianKDEstimator(R_uu[:, None], bw=bw),
        'dd': GaussianKDEstimator(R_dd[:, None], bw=bw),
        'ud': GaussianKDEstimator(R_ud[:, None], bw=bw),
        'decorr': GaussianKDEstimator(R_decorr[:, None], bw=bw),
    }
