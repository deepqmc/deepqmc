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
            R (~jax.Array): the nuclei positions of the molecular configuration.

        Returns:
            ~deepqmc.types.SamplerState:
            the sampler state holding electron positions and data about the sampler trajectory.
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
            R (~jax.Array): the nuclei positions of the molecular configuration.

        Returns:
            tuple[~deepqmc.types.SamplerState, ~deepqmc.types.PhysicalConfiguration, ~deepqmc.types.Stats]:
            the new sampler state, a physical configuration and statistics about the sampling trajectory.
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
            R (~jax.Array): the nuclei positions of the molecular configuration.

        Returns:
            ~deepqmc.types.SamplerState: the updated sampler state holding electron positions and data about the sampler trajectory.
        """
        ...


class NucleiSampler(Protocol):
    r"""Protocol for nuclear geometry samplers.

    :class:`~deepqmc.sampling.base.NucleiSampler` objects implement samplers for the
    nuclear coordinates, used during transferable training across multiple molecular
    geometries. The interface mirrors
    :class:`~deepqmc.sampling.base.ElectronSampler` but operates on nuclear positions
    rather than electron positions. Nuclei samplers are not using energy based
    accept and reject criteria.
    """

    def init(self, nuc_coords: jax.Array) -> SamplerState:
        r"""Initialize the nuclear sampler state.

        Args:
            nuc_coords (~jax.Array): initial nuclear coordinates of shape
                ``(n_nuc, 3)``.

        Returns:
            ~deepqmc.types.SamplerState: the initial sampler state.
        """
        ...

    def sample(
        self, rng: KeyArray, state: SamplerState
    ) -> tuple[SamplerState, jax.Array, Stats]:
        r"""Propose a new set of nuclear coordinates.

        Args:
            rng (~deepqmc.types.KeyArray): an rng key for the coordinate proposal.
            state (~deepqmc.types.SamplerState): the current sampler state.

        Returns:
            tuple[~deepqmc.types.SamplerState, ~jax.Array, ~deepqmc.types.Stats]: the
                updated sampler state, the proposed nuclear coordinates and sampling statistics.
        """
        ...


class ElectronWarp(Protocol):
    r"""Protocol for electron warp functions.

    An :class:`~deepqmc.sampling.base.ElectronWarp` displaces the electron positions
    stored inside a sampler state in response to a change in nuclear geometry.
    Applying a warp before
    re-equilibrating the sampler avoids large acceptance-rate drops when nuclear
    coordinates move during optimization or potential energy exploration.
    """

    def __call__(
        self, rng: KeyArray, R: jax.Array, dR: jax.Array, smpl_state: SamplerState
    ) -> SamplerState:
        r"""Apply the electron warp.

        Args:
            rng (~deepqmc.types.KeyArray): an rng key for stochastic warps.
            R (~jax.Array): the new nuclear coordinates.
            dR (~jax.Array): the nuclear displacement ``R_new - R_old``.
            smpl_state (~deepqmc.types.SamplerState): the current sampler state whose
                electron positions are to be warped.

        Returns:
            ~deepqmc.types.SamplerState: the sampler state with warped electron
                positions.
        """
        ...
