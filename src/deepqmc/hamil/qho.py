import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from ..physics import laplacian
from ..types import PhysicalConfiguration
from .base import Hamiltonian

__all__ = ['QHOHamiltonian']


class QHOHamiltonian(Hamiltonian):
    r"""Hamiltonian for the quantum harmonic oscillator."""

    def __init__(self, dim, mass, nu):
        self.dim = (dim,)
        self.mass = mass
        self.nu = nu

    def local_energy(self, wf):
        def loc_ene(rng, phys_conf):
            def wave_function(r):
                return wf(jdc.replace(phys_conf, r=r)).log

            pot = 1 / 2 * self.mass * self.nu**2 * jnp.sum(phys_conf.r**2)
            lap_log, grad_log = laplacian(wave_function)(phys_conf.r)
            kin = -1 / (2 * self.mass) * (lap_log + jnp.sum(grad_log**2))
            return kin + pot, {}

        return loc_ene

    def init_sample(self, rng, Rs, n):
        return PhysicalConfiguration(
            jnp.zeros((n, *self.dim)),
            jax.random.normal(rng, (n, *self.dim)),
            jnp.zeros(n, dtype=jnp.int32),
        )

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
