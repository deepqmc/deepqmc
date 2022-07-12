import jax.numpy as jnp

from ..utils import laplacian

__all__ = ()


class QHOHamiltonian:
    def __init__(self, dim, mass, nu):
        self.dim = dim
        self.mass = mass
        self.nu = nu

    def local_energy(self, wf):
        def loc_ene(r):
            pot = 1 / 2 * self.mass * self.nu**2 * jnp.sum(r**2)
            lap_log, grad_log = laplacian(lambda x: wf(x).log)(r)
            kin = -1 / (2 * self.mass) * (lap_log + jnp.sum(grad_log**2))
            return kin + pot

        return loc_ene
