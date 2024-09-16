from typing import Protocol

import jax

from deepqmc.types import (
    KeyArray,
    Params,
    PhysicalConfiguration,
    SamplerState,
    Stats,
)


class ElectronSampler(Protocol):
    r"""Protocol for :class:`~deepqmc.sampling.base.ElectronSampler` objects.

    :class:`~deepqmc.sampling.base.ElectronSampler` objects implement Markov chain
    samplers for the electron positions. The samplers are assumed to implement a batch
    of walkers for a single electronic state on a single molecule and may be vmapped
    to fit the respective context they are used in. Electron samplers can be combined
    with :func:`~deepqmc.sampling.chain`.
    """

    def init(self, rng: KeyArray, params: Params, n: int, R: jax.Array) -> SamplerState:
        r"""
        Initializes the sampler state.

        Args:
            rng (~deepqmc.types.KeyArray): an rng key for the initialization of electron
                positions.
            params (~deepqmc.types.Params): the parameters of the wave function that is
                being sampled.
            n (int): the number of walkers to propagate in parallel.
            R (jax.Array): the nuclei positions of the molecular configuration.

        Returns:
            :type:`~deepqmc.types.SamplerState`: the sampler state holding electron
            positions and data about the sampler trajectory.
        """
        ...

    def sample(
        self, rng: KeyArray, state: SamplerState, params: Params, R: jax.Array
    ) -> tuple[SamplerState, PhysicalConfiguration, Stats]:
        r"""
        Propagates the sampler state.

        Args:
            rng (~deepqmc.types.KeyArray): an rng key for the proposal of electron
                positions.
            state (~deepqmc.types.SamplerState): the state of the sampler from the
                previous step.
            params (~deepqmc.types.Params): the parameters of the wave function that is
                being sampled.
            R (jax.Array): the nuclei positions of the molecular configuration.

        Returns:
            tuple[:type:`~deepqmc.types.SamplerState`,
            :type:`~deepqmc.types.PhysicalConfiguration`, :type:`~deepqmc.types.Stats`]:
            the new sampler state, a physical configuration and statistics about the
            sampling trajectory.
        """
        ...

    def update(self, state: SamplerState, params: Params, R: jax.Array) -> SamplerState:
        r"""
        Updates the sampler state.

        The sampler state is updated to account for changes in the wave function due
        to a parameter update.

        Args:
            state (~deepqmc.types.SamplerState): the state of the sampler before
                parameter update.
            params (~deepqmc.types.Params): the new parameters of the wave function.
            R (jax.Array): the nuclei positions of the molecular configuration.

        Returns:
            :type:`~deepqmc.types.SamplerState`: the updated sampler state holding
            electron positions and data about the sampler trajectory.
        """
        ...


class NucleiSampler(Protocol):
    r"""Protocol for nuclei samplers."""

    def init(self, nuc_coords: jax.Array) -> SamplerState: ...

    def sample(
        self, rng: KeyArray, state: SamplerState
    ) -> tuple[SamplerState, jax.Array, Stats]: ...


class ElectronWarp(Protocol):
    r"""Protocol for electron warps."""

    def __call__(
        self, rng: KeyArray, R: jax.Array, dR: jax.Array, smpl_state: SamplerState
    ) -> SamplerState: ...
