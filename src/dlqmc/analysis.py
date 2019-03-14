import numpy as np


def blocking(x, max_B=None):
    N = len(x)
    sigmas = []
    sigma = x.std()
    max_B = max_B or int(np.log2(N))
    for log_B in range(0, max_B):
        B = 2 ** log_B
        sigma_B = (
            x[-(N // B * B) :].view(-1, B).mean(dim=-1).std().item()
            * np.sqrt(B)
            / sigma
        )
        sigmas.append((log_B, sigma_B))
    return sigmas


def autocorr_coeff(ks, x):
    Cs = []
    x_mean = x.mean()
    var = ((x - x_mean) ** 2).sum()
    for k in ks:
        end = -k or x.numel()
        autocov = ((x[:end] - x_mean) * (x[k:] - x_mean)).sum()
        C = autocov / var
        Cs.append(C.item())
    return Cs
