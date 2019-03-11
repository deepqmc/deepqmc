import numpy as np


def blocking(x, maxB=None):
    N = len(x)
    y = []
    sigma = x.std()
    maxB = maxB or int(np.log2(N))
    for logB in range(0, maxB):
        B = 2 ** logB
        y.append(
            (
                logB,
                x[-(N // B * B) :].view(-1, B).mean(dim=-1).std().item()
                * np.sqrt(B)
                / sigma,
            )
        )
    return y


def autocorr(ks, x):
    Cs = []
    x_mean = x.mean()
    for k in ks:
        end = -k or x.numel()
        C = ((x[:end] - x_mean) * (x[k:] - x_mean)).sum() / (
            (x[:end] - x_mean) ** 2
        ).sum()
        Cs.append(C.item())
    return Cs
