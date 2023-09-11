import haiku as hk
import jax.numpy as jnp


class DeepQMCCusp:
    r"""Compute the DeepQMC cusp factor.

    Computes the factor:
    :math:`-\frac{\text{scale}}{\sum_{i<j}\alpha * (1 + \alpha r_{ij})}`, where
    :math:`r_{ij}` are the electron-electron or electron-nuclei distances.
    """

    def __call__(self, scale, alpha, dist):
        return -(scale / (alpha * (1 + alpha * dist))).sum()


class PsiformerCusp:
    r"""Compute the Psiformer cusp factor.

    Computes the factor:
    :math:`-\frac{\text{scale}}{\sum_{i<j}\alpha^2 * (\alpha + r_{ij})}`, where
    :math:`r_{ij}` are the electron-electron or electron-nuclei distances.
    """

    def __call__(self, scale, alpha, dist):
        return -((scale * alpha**2) / (alpha + dist)).sum()


class CuspAsymptotic(hk.Module):
    """Base class for nuclear and electronic cusps."""

    def __init__(self, *, cusp_function, trainable_alpha):
        super().__init__()
        self.trainable_alpha = trainable_alpha
        self.cusp_function = cusp_function

    def get_alpha(self, value, name):
        return (
            hk.get_parameter(
                f'{name}_alpha',
                (),
                init=lambda s, d: jnp.array(value, dtype=d).reshape(s),
            )
            if self.trainable_alpha
            else jnp.array(value, dtype=float)
        )


class ElectronicCuspAsymptotic(CuspAsymptotic):
    r"""Calculate a multiplicative factor, that implements the electronic cusps.

    Args:
        same_scale (float): scaling factor to use for same spin electron cusps.
        anti_scale (float): scaling factor to use for anti spin electron cusps.
        alpha (float): default 1, the :math:`\alpha` parameter in the above
            cusp equations.
        trainable_alpha (bool): whether the :math:`\alpha` is trainable
        cusp_function (Callable): an instance of either
            :class:`deepqmc.wf.nn_wave_function.cusp.DeepQMCCusp` or
            :class:`deepqmc.wf.nn_wave_function.cusp.PsiformerCusp`.
    """

    def __init__(self, *, same_scale, anti_scale, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.same_scale, self.anti_scale = same_scale, anti_scale
        self.same_alpha = self.get_alpha(alpha, 'same')
        self.anti_alpha = self.get_alpha(alpha, 'anti')

    def __call__(self, same_dists, anti_dists):
        return self.cusp_function(
            self.same_scale, self.same_alpha, same_dists
        ) + self.cusp_function(self.anti_scale, self.anti_alpha, anti_dists)


class NuclearCuspAsymptotic(CuspAsymptotic):
    r"""Calculate a multiplicative factor, that implements the nuclear cusps.

    Args:
        nuclear_charges ([float]): the array of nuclear charges of the molecule
        alpha (float): default 1, the :math:`\alpha` parameter in the above
            cusp equations.
        trainable_alpha (bool): whether the :math:`\alpha` is trainable
        cusp_function (Callable): an instance of either
            :class:`deepqmc.wf.nn_wave_function.cusp.DeepQMCCusp` or
            :class:`deepqmc.wf.nn_wave_function.cusp.PsiformerCusp`.
    """

    def __init__(self, nuclear_charges, *, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.nuclear_charges = nuclear_charges[None]  # [1, n_nuclei]
        self.alpha = self.get_alpha(alpha, 'nuc')

    def __call__(self, dists):
        # dists: [n_elec, n_nuc]
        return self.cusp_function(self.nuclear_charges, self.alpha, dists)
