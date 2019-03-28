import numpy as np
import torch


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
