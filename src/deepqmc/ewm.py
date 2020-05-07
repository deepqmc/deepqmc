import logging
import math

import numpy as np
from uncertainties import ufloat

log = logging.getLogger(__name__)


class EWMElocMonitor:
    _PERCENTILES = (
        100
        * (1 + np.array([math.erf(x / math.sqrt(2)) for x in [0, -3, -2, -1, 1, 2, 3]]))
        / 2
    )
    _LABELS = 'med -3s -2s -1s +1s +2s +3s mean mean_slow'.split()

    def __init__(self):
        self.step = 0
        self.blowup_detection = {}

    @property
    def energy(self):
        return ufloat(self.ewm_mean[-1], np.sqrt(self.ewm_err[-1]))

    def update(self, E_loc):
        stat = np.empty(len(self._LABELS))
        a = np.empty_like(stat)
        stat[:-2] = np.percentile(E_loc, self._PERCENTILES)
        stat[-2:] = E_loc.mean()
        a[:-1] = min(0.96, 1 - 1 / (2 + self.step / 10))
        a[-1] = min(0.99, 1 - 1 / (2 + self.step / 10))
        is_outlier = (
            np.abs(stat - self.ewm_mean) > 3 * np.sqrt(self.ewm_var)
            if self.step > 5
            else np.zeros_like(stat, dtype=bool)
        )
        if is_outlier[:8].sum() >= 5:
            log.info(f'Detected EWM outlier: step {self.step}')
            if not self.blowup_detection:
                self.blowup_detection = {
                    'init': self.step,
                    'step': self.step,
                    'start': self.ewm_mean[-2],
                }
            else:
                self.blowup_detection['step'] = self.step
        if self.step == 0:
            self.ewm_mean = stat.copy()
            self.ewm_var = np.zeros_like(stat)
            self.ewm_err = np.zeros_like(stat)
        else:
            ewm_var = (1 - a) * (stat - self.ewm_mean) ** 2 + a * self.ewm_var
            ewm_mean = (1 - a) * stat + a * self.ewm_mean
            ewm_err = (1 - a) ** 2 * self.ewm_var + a ** 2 * self.ewm_err
            self.ewm_var = np.where(is_outlier, self.ewm_var, ewm_var)
            self.ewm_mean = np.where(is_outlier, self.ewm_mean, ewm_mean)
            self.ewm_err = np.where(is_outlier, self.ewm_err, ewm_err)
        if self.blowup_detection and self.step - self.blowup_detection['step'] > 50:
            self.blowup_detection = {}
        if self.blowup_detection:
            self.blowup_detection['indicator'] = (
                self.ewm_mean[-2] - self.blowup_detection['start']
            ) / np.sqrt(self.ewm_var[-2])
        self.step += 1
