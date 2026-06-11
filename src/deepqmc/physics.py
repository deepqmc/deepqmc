from collections.abc import Callable
from functools import partial
from typing import Optional, Protocol

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from .geom.general import pairwise_distance, pairwise_self_distance
from .types import (
    Energy,
    KeyArray,
    ParametrizedWaveFunction,
    Params,
    PhysicalConfiguration,
    Psi,
    WaveFunction,
)
from .utils import triu_flat

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
            phys_conf (:class:`~deepqmc.types.PhysicalConfiguration`): electron and
                nuclear coordinates.

        Returns:
            ~deepqmc.types.Energy: the local potential energy.
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

        Args:
            rng (Optional[:class:`~deepqmc.types.KeyArray`]): PRNG key, or None.
            phys_conf (:class:`~deepqmc.types.PhysicalConfiguration`): electron and
                nuclear coordinates.
            wf (~deepqmc.types.WaveFunction): wave function.

        Returns:
            ~deepqmc.types.Energy: the non-local contribution to the energy.
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
            phys_conf (:class:`~deepqmc.types.PhysicalConfiguration`): electron and
                nuclear coordinates.
            wf (:class:`~deepqmc.types.WaveFunction`): wave function.
            laplacian_factory (Callable): factory to compute the laplacian and gradient.

        Returns:
            tuple[~deepqmc.types.Energy, jax.Array, jax.Array]: the kinetic energy,
            the laplacian of the log WF, and the squared quantum force.
        """

        def wave_function(r: jax.Array) -> jax.Array:
            pc = jdc.replace(phys_conf, r=r.reshape(-1, 3))
            return wf(pc).log

        lap_log_psis, quantum_force = laplacian_factory(wave_function)(
            phys_conf.r.flatten()
        )
        Es_kin = -0.5 * (lap_log_psis + (quantum_force**2).sum(axis=-1))
        return Es_kin, lap_log_psis, (quantum_force**2).sum(axis=-1)


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


def reverse_forward_laplacian(
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
    hamil, parametrized_wf: ParametrizedWaveFunction
) -> Callable[[Params, PhysicalConfiguration], jax.Array]:
    """Returns a function to evaluate the spin expectation value (s^2)."""

    up_minus_down = hamil.n_up - hamil.n_down

    def evaluate_spin_(params: Params, phys_conf: PhysicalConfiguration) -> jax.Array:
        s2 = up_minus_down / 2 * (up_minus_down / 2 + 1) + hamil.n_down
        wf = partial(parametrized_wf, params)

        orig_psi = wf(phys_conf)
        permute_single_down_with_all_up = make_permute_single_down_with_all_up(
            wf, phys_conf, orig_psi, hamil.n_up
        )

        s2 = jax.lax.fori_loop(
            hamil.n_up, hamil.n_up + hamil.n_down, permute_single_down_with_all_up, s2
        )

        return s2

    return evaluate_spin_


def make_permute_single_down_with_all_up(
    wf: WaveFunction, phys_conf: PhysicalConfiguration, orig_psi: Psi, n_up: int
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    r"""Return a function that accumulates permuted wave function ratios.

    The returned function computes
    :math:`- \sum_{\alpha} \frac{\hat P_{\alpha\beta} \Psi}{\Psi}`,
    for one value of :math:`beta`.
    """

    def permute_single_down_with_all_up(
        down_idx: jax.Array, outer_accumulator: jax.Array
    ) -> jax.Array:
        original_electron_indices = jnp.arange(phys_conf.r.shape[0])

        def permute_single_down_with_single_up(
            up_idx: jax.Array, inner_accumulator: jax.Array
        ) -> jax.Array:
            permuted_electron_indices = (
                original_electron_indices.at[down_idx]
                .set(up_idx)
                .at[up_idx]
                .set(down_idx)
            )
            permuted_phys_conf = jdc.replace(
                phys_conf, r=phys_conf.r[permuted_electron_indices]
            )
            permuted_psi = wf(permuted_phys_conf)
            inner_accumulator -= (
                orig_psi.sign
                * permuted_psi.sign
                * jnp.exp(permuted_psi.log - orig_psi.log)
            )
            return inner_accumulator

        return jax.lax.fori_loop(
            0, n_up, permute_single_down_with_single_up, outer_accumulator
        )

    return permute_single_down_with_all_up


def make_stochastic_spin_raising_operator(
    hamil, parametrized_wf: ParametrizedWaveFunction
):
    def evaluate_stochastic_spin_raising_operator(
        params: Params, phys_conf: PhysicalConfiguration, down_idx: jax.Array
    ):
        wf = partial(parametrized_wf, params)
        orig_psi = wf(phys_conf)
        permute_single_down_with_all_up = make_permute_single_down_with_all_up(
            wf, phys_conf, orig_psi, hamil.n_up
        )
        return permute_single_down_with_all_up(down_idx, jnp.array(1.0))

    return evaluate_stochastic_spin_raising_operator
