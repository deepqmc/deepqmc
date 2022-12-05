import jax
import jax.numpy as jnp

from ..physics import laplacian
from .base import Hamiltonian

__all__ = ['QHOHamiltonian']


class QHOHamiltonian(Hamiltonian):
    r"""Hamiltonian for the quantum harmonic oscillator."""

    def __init__(self, dim, mass, nu):
        self.dim = (dim,)
        self.mass = mass
        self.nu = nu

    def local_energy(self, wf):
        def loc_ene(state, r):
            pot = 1 / 2 * self.mass * self.nu**2 * jnp.sum(r**2)
            lap_log, grad_log = laplacian(lambda x: wf(state, x).log)(r)
            kin = -1 / (2 * self.mass) * (lap_log + jnp.sum(grad_log**2))
            return kin + pot, {}

        return loc_ene

    def init_sample(self, rng, n):
        return jax.random.normal(rng, (n, *self.dim))

    def stats(self, r):
        mean = jnp.mean(r, axis=0, keepdims=True)
        sd = jnp.sqrt(
            jnp.mean(
                jnp.sum((r - mean) ** 2, axis=-1, keepdims=True), axis=0, keepdims=True
            )
        )
        return {
            'r/mean': jnp.linalg.norm(mean).item(),
            'r/std': sd.item(),
        }
