import matplotlib as mpl
import matplotlib.scale
import matplotlib.ticker
import matplotlib.transforms
import numpy as np

__all__ = ()


def corr_ene_tf(a):
    with np.errstate(divide='ignore', invalid='ignore'):
        out = -np.log10(1 - a)
        out = np.where(a >= 1, 10, out)
        return out


def corr_ene_inv_tf(a):
    return 1 - 10 ** (-a)


class CorrelationEnergyTransform(mpl.transforms.Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def transform_non_affine(self, a):
        return corr_ene_tf(a)

    def inverted(self):
        return InvertedCorrelationEnergyTransform()


class InvertedCorrelationEnergyTransform(mpl.transforms.Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def transform_non_affine(self, a):
        return corr_ene_inv_tf(a)

    def inverted(self):
        return CorrelationEnergyTransform()


class CorrelationEnergyLocator(mpl.ticker.Locator):
    def __init__(self, subs=1):
        self.subs = subs

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        vmin = np.floor(corr_ene_tf(vmin))
        vmax = np.ceil(corr_ene_tf(vmax))
        bases = np.arange(vmin, vmax + 1e-10)
        decades = corr_ene_inv_tf(bases)
        ticks = np.concatenate(
            [
                np.arange(decades[i], decades[i + 1], 10 ** -bases[i] / self.subs)
                for i in range(len(decades) - 1)
            ]
        )
        return ticks

    def view_limits(self, vmin, vmax):
        lims = corr_ene_tf(np.array([vmin, vmax]))
        rng = lims[1] - lims[0]
        lims = np.array([lims[0] - 0.05 * rng, lims[1] + 0.05 * rng])
        return tuple(corr_ene_inv_tf(lims))


class CorrelationEnergyFormatter(mpl.ticker.Formatter):
    def __call__(self, x, pos=None):
        acc = max(0, int(np.round(corr_ene_tf(x))) - 2)
        return f'{100 * x:.{acc}f}%'


class CorrelationEnegryScale(mpl.scale.ScaleBase):
    name = 'corr_energy'

    def __init__(self, axis, subs=10):
        self.subs = subs

    def get_transform(self):
        return CorrelationEnergyTransform()

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(CorrelationEnergyLocator())
        axis.set_minor_locator(CorrelationEnergyLocator(self.subs))
        axis.set_major_formatter(CorrelationEnergyFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return min(vmin, 1 - 1e-10), min(vmax, 1 - 1e-10)


mpl.scale.register_scale(CorrelationEnegryScale)
