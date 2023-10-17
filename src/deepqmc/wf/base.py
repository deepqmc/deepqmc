import logging
import operator

import haiku as hk
import jax
import jax.numpy as jnp

from deepqmc.parallel import replicate_on_devices

__all__ = []

log = logging.getLogger(__name__)


def init_wf_params(rng, hamil, ansatz):
    rng_sample, rng_params = jax.random.split(rng)
    try:
        # QC
        R_shape = (len(hamil.mol.charges), 3)
    except AttributeError:
        # QHO
        R_shape = 0
    phys_conf = hamil.init_sample(rng_sample, jnp.zeros(R_shape), 1)[0]
    params = ansatz.init(rng_params, phys_conf)

    num_params = jax.tree_util.tree_reduce(
        operator.add, jax.tree_util.tree_map(lambda x: x.size, params)
    )
    log.info(f'Number of model parameters: {num_params}')
    params = replicate_on_devices(params)
    return params


class WaveFunction(hk.Module):
    r"""
    Base class for all trial wave functions.

    Shape:
        - Input, :math:`\mathbf r`, (float, :math:`(N,3)`, a.u.): particle
            coordinates
        - Output1, :math:`\ln|\psi(\mathbf r)|` (float):
        - Output2, :math:`\operatorname{sgn}\psi(\mathbf r)` (float):
    """

    def __init__(self, hamil):
        super().__init__()
        self.mol = hamil.mol
        self.n_up, self.n_down = hamil.n_up, hamil.n_down

    @property
    def spin_slices(self):
        return slice(None, self.n_up), slice(self.n_up, None)

    def forward(self, rs):
        return NotImplemented
