import haiku as hk
import jax
import jax.numpy as jnp

__all__ = []


def init_wf_params(rng, hamil, ansatz):
    rng_sample, rng_params = jax.random.split(rng)
    try:
        # QC
        R_shape = (len(hamil.mol.charges), 3)
    except AttributeError:
        # QHO
        R_shape = 0
    phys_conf = hamil.init_sample(rng_sample, jnp.zeros(R_shape), 1)[0]
    params, _ = ansatz.init(rng_params, phys_conf)
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

    def __init__(self, mol):
        super().__init__()
        self.mol = mol
        self.n_up, self.n_down = mol.n_up, mol.n_down

    @property
    def spin_slices(self):
        return slice(None, self.n_up), slice(self.n_up, None)

    def forward(self, rs):
        return NotImplemented
