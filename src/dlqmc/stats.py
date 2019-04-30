import numpy as np
import torch

from .utils import batch_eval


class GaussianKDEstimator:
    def __init__(self, xs, max_memory=1.0, weights=None, *, bw):
        assert len(xs.shape) == 2
        self._xs = xs
        self._weights = weights if weights is not None else None
        self._width = np.sqrt(2) * bw
        self._bs = int(2 ** 30 * max_memory) // (xs.nelement() * xs.element_size())

    def __call__(self, xs):
        assert len(xs.shape) == 2
        if len(xs) > self._bs:
            return batch_eval(self, xs.split(self._bs))
        kernel = ((xs[:, None] - self._xs) ** 2).sum(dim=-1) / self._width ** 2
        norm = 1 / (len(self._xs) * (np.sqrt(np.pi) * self._width) ** xs.shape[1])
        basis = torch.exp(-kernel)
        if self._weights is not None:
            basis = self._weights * basis
        return norm * basis.sum(dim=-1)
