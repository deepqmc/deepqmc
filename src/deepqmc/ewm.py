import logging
import math

import numpy as np
from uncertainties import ufloat, unumpy as unp

__all__ = ()

log = logging.getLogger(__name__)


class EWMAverage:
    def __init__(
        self, init=5, outlier=3, outlier_maxlen=3, max_alpha=0.999, decay_alpha=10
    ):
        self.step = 0
        self._init = init
        self._outlier = outlier
        self._outlier_maxlen = outlier_maxlen
        self._max_alpha = max_alpha
        self._decay_alpha = decay_alpha

    def _alpha(self, n):
        return min(self._max_alpha, 1 - 1 / (2 + n / self._decay_alpha))

    @property
    def mean(self):
        return unp.uarray(self._mean, np.sqrt(self._sqerr))

    @property
    def var(self):
        return self._var

    @property
    def std(self):
        return np.sqrt(self._var)

    def update(self, x, alpha=None):
        x = np.array(x)
        a = alpha if alpha is not None else self._alpha(self.step)
        is_outlier = (
            (np.abs(x - self._mean) > self._outlier * np.sqrt(self._var))
            & (self._n_outlier <= self._outlier_maxlen)
            if self.step >= self._init
            else np.zeros_like(x, dtype=bool)
        )
        no_update = is_outlier | np.isnan(x)
        if self.step == 0:
            self._mean = x.copy()
            self._var = np.zeros_like(x)
            self._sqerr = np.zeros_like(x)
            self._n_outlier = np.zeros_like(x)
        else:
            var = (1 - a) * (x - self._mean) ** 2 + a * self._var
            mean = (1 - a) * x + a * self._mean
            sqerr = (1 - a) ** 2 * self._var + a ** 2 * self._sqerr
            self._var = np.where(no_update, self._var, var)
            self._mean = np.where(no_update, self._mean, mean)
            self._sqerr = np.where(no_update, self._sqerr, sqerr)
            self._n_outlier = np.where(is_outlier, self._n_outlier + 1, 0)
        self.step += 1
        return is_outlier


class EWMMonitor(EWMAverage):
    I = '-3s -2s -1s med +1s +2s +3s mean mean_slow'.split()
    I = {l: i for i, l in enumerate(I)}

    def __init__(self, stat_outlier=6, blowup_maxlen=25, blowup_thre=0.5, **kwargs):
        super().__init__(max_alpha=1, **kwargs)
        self.blowup = {}
        self._stat_outlier = stat_outlier
        self._blowup_maxlen = blowup_maxlen
        self._blowup_thre = blowup_thre
        percentiles = [math.erf(x / math.sqrt(2)) for x in range(-3, 4)]
        self._PERCENTILES = 100 * (1 + np.array(percentiles)) / 2

    def mean_of(self, label):
        i = self.I[label]
        return ufloat(self._mean[i], np.sqrt(self._sqerr[i]))

    def update(self, x):
        I = self.I
        stat = np.empty(len(self.I))
        a = np.empty_like(stat)
        stat[: len(self._PERCENTILES)] = np.percentile(x, self._PERCENTILES)
        stat[I['mean'] :] = x.mean()
        alpha = self._alpha(self.step)
        a[: I['mean_slow']] = min(0.96, alpha)
        a[I['mean_slow']] = min(0.999, alpha)
        is_outlier = super().update(stat, a)
        if is_outlier[: I['mean_slow']].sum() >= self._stat_outlier:
            if not self.blowup:
                self.blowup = {
                    'init': self.step,
                    'step': self.step,
                    'start': self._mean[I['mean']],
                }
            else:
                self.blowup['step'] = self.step
        if self.blowup and self.step - self.blowup['step'] > self._blowup_maxlen:
            self.blowup = {}
        if self.blowup:
            self.blowup['indicator'] = (
                self._mean[I['mean']] - self.blowup['start']
            ) / np.sqrt(self._var[I['mean']])
            self.blowup['in_blowup'] = self.blowup['indicator'] > self._blowup_thre
        return is_outlier, stat
