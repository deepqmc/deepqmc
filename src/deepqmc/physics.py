from collections.abc import Callable
from typing import Optional, Protocol

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from .types import (
    Energy,
    KeyArray,
    ParametrizedWaveFunction,
    Params,
    PhysicalConfiguration,
    WaveFunction,
)
from .utils import norm, triu_flat

__all__ = ()


class LaplacianFactory(Protocol):
    r"""Protocol class for Laplacian factories.

    A Laplacian factory takes as input a function and returns a function that
    computes the laplacian and gradient of the input function
    """

    def __call__(
        self, f: Callable[[jax.Array], jax.Array]
    ) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]: ...


class Potential(Protocol):
    r"""Protocol for :class:`~deepqmc.types.Potential` objects.

    Implements the (effective core) potential in which the electrons move. Does not
    include the electron-electron repulsion.
    """

    def local_potential(self, phys_conf: PhysicalConfiguration) -> Energy:
        r"""Compute the (local effective core) potential energy of the electrons.

        Args:
            phys_conf (:class:`deepqmc.types.PhysicalConfiguration`): electron and
                nuclear coordinates.
        """
        ...

    def nonloc_potential(
        self,
        rng: Optional[KeyArray],
        phys_conf: PhysicalConfiguration,
        wf: WaveFunction,
    ) -> Energy:
        r"""Compute the non-local potential energy.

        When the potential is fully local, (e.g. Coulomb potential or
        PseudoHamiltonian), this function should return 0.0.
        """
        return jnp.array(0.0)

    def kinetic_term(
        self,
        phys_conf: PhysicalConfiguration,
        wf: WaveFunction,
        laplacian_factory: LaplacianFactory,
    ) -> tuple[Energy, jax.Array, jax.Array]:
        r"""Compute the kinetic term of the Hamiltonian.

        Typically, -1/2Δ, where Δ is the laplacian of the wave function.

        Args:
            phys_conf (:class:`deepqmc.types.PhysicalConfiguration`): electron and
                nuclear coordinates.
            wf (:class:`deepqmc.types.WaveFunction`): wave function.
            laplacian_factory (Callable): factory to compute the laplacian and gradient.
        """

        def wave_function(r: jax.Array) -> jax.Array:
            pc = jdc.replace(phys_conf, r=r.reshape(-1, 3))
            return wf(pc).log

        lap_log_psis, quantum_force = laplacian_factory(wave_function)(
            phys_conf.r.flatten()
        )
        Es_kin = -0.5 * (lap_log_psis + (quantum_force**2).sum(axis=-1))
        return Es_kin, lap_log_psis, (quantum_force**2).sum(axis=-1)


def pairwise_distance(coords1: jax.Array, coords2: jax.Array) -> jax.Array:
    return jnp.linalg.norm(coords1[..., :, None, :] - coords2[..., None, :, :], axis=-1)


def pairwise_diffs(coords1: jax.Array, coords2: jax.Array) -> jax.Array:
    diffs = coords1[..., :, None, :] - coords2[..., None, :, :]
    return jnp.concatenate([diffs, (diffs**2).sum(axis=-1, keepdims=True)], axis=-1)


def pairwise_self_distance(coords: jax.Array, full: bool = False) -> jax.Array:
    i, j = jnp.triu_indices(coords.shape[-2], k=1)
    diffs = coords[..., :, None, :] - coords[..., None, :, :]
    dists = norm(diffs[..., i, j, :], safe=True, axis=-1)
    if full:
        dists = (
            jnp.zeros(diffs.shape[:-1])
            .at[..., i, j]
            .set(dists)
            .at[..., j, i]
            .set(dists)
        )
    return dists


def nuclear_energy(phys_conf: PhysicalConfiguration, ns_valence: jax.Array) -> Energy:
    coulombs = triu_flat(ns_valence[:, None] * ns_valence) / pairwise_self_distance(
        phys_conf.R
    )
    return coulombs.sum()


def electronic_potential(phys_conf: PhysicalConfiguration) -> Energy:
    dists = pairwise_self_distance(phys_conf.r)
    return (1 / dists).sum(axis=-1)


class NuclearCoulombPotential(Potential):
    """Class for the classical Coulomb potential."""

    def __init__(self, charges: jax.Array):
        self.charges = charges
        self.ns_valence = charges

    def local_potential(self, phys_conf: PhysicalConfiguration) -> Energy:
        dists = pairwise_distance(phys_conf.r, phys_conf.R)
        return -(self.charges / dists).sum(axis=(-1, -2))

    def nonloc_potential(
        self,
        rng: Optional[KeyArray],
        phys_conf: PhysicalConfiguration,
        wf: WaveFunction,
    ) -> Energy:
        return jnp.array(0.0)


def laplacian(
    f: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    def lap(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        n_coord = len(x)
        grad_f = jax.grad(f)
        df, grad_f_jvp = jax.linearize(grad_f, x)
        eye = jnp.eye(n_coord)
        d2f = lambda i, val: val + grad_f_jvp(eye[i])[i]
        d2f_sum = jax.lax.fori_loop(0, n_coord, d2f, 0.0)
        return d2f_sum, df

    return lap


def evaluate_spin(
    hamil, wf: ParametrizedWaveFunction
) -> Callable[[Params, PhysicalConfiguration], jax.Array]:
    """Returns a function to evaluate the spin expectation value (s^2)."""
    nspins = (hamil.n_up, hamil.n_down)

    def evaluate_spin_(params: Params, phys_conf: PhysicalConfiguration) -> jax.Array:
        na, nb = sorted(nspins, reverse=True)
        s2 = (na - nb) / 2 * ((na - nb) / 2 + 1) + nb

        psi = wf(params, phys_conf)
        r_up, r_down = jnp.split(phys_conf.r, nspins[:1], axis=-2)

        def _inner(j, val):
            i, s2 = val
            r_perm = jnp.concatenate(
                (r_up.at[i].set(r_down[j]), r_down.at[j].set(r_up[i]))
            )
            psi_perm = wf(params, jdc.replace(phys_conf, r=r_perm))
            s2 -= psi.sign * psi_perm.sign * jnp.exp(psi_perm.log - psi.log)
            return i, s2

        def _outer(i, s2):
            return jax.lax.fori_loop(0, nspins[1], _inner, (i, s2))[1]

        s2 = jax.lax.fori_loop(0, nspins[0], _outer, s2)
        return s2

    return evaluate_spin_


def coulomb_force(
    r1: jax.Array,
    r2: jax.Array,
    c1: jax.Array,
    c2: jax.Array,
    remove_self_int: bool = False,
) -> jax.Array:
    dists = r1[:, None] - r2[None]
    force = (
        (c1[:, None] * c2[None])[..., None]
        * dists
        / jnp.linalg.norm(dists, axis=-1, keepdims=True) ** 3
    )
    if remove_self_int:
        force = force.at[jnp.arange(len(r1)), jnp.arange(len(r2))].set(0)
    return force.sum(-2)
