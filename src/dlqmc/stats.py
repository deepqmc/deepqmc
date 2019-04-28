import numpy as np
import torch


class GaussianKDEstimator:
    def __init__(self, xs, max_memory=1.0, *, bw):
        assert len(xs.shape) == 2
        self._xs = xs
        self._width = np.sqrt(2) * bw
        self._n_pts = len(xs)
        self._bs = int(2 ** 30 * max_memory) // (xs.nelement() * xs.element_size())

    def __call__(self, xs):
        assert len(xs.shape) == 2
        if len(xs) > self._bs:
            return torch.cat([self(xs_batch) for xs_batch in xs.split(self._bs)])
        kernel = ((xs[:, None] - self._xs) ** 2).sum(dim=-1) / self._width ** 2
        norm = 1 / (self._n_pts * np.sqrt(np.pi) * self._width)
        return norm * torch.exp(-kernel).sum(dim=-1)
