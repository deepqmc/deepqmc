from __future__ import annotations

from typing import Optional, Protocol

import jax

from ..hamil import MolecularHamiltonian
from ..types import Ansatz, Batch, Energy, KeyArray, Params, Stats

__all__ = ['LossFunction', 'LossFunctionFactory', 'LossAndGradFunction']


class LossFunction(Protocol):
    r"""Protocol for loss functions used during wave function training.

    A :class:`LossFunction` takes model parameters, an RNG key, and a batch of
    electron configurations and returns a scalar loss value together with
    auxiliary per-sample data.
    """

    def __call__(
        self,
        params: list[Params],
        rng: KeyArray,
        batch: Batch,
    ) -> tuple[jax.Array, tuple[Optional[Energy], Optional[jax.Array], Stats]]:
        r"""Evaluate the loss function.

        Args:
            params (list[~deepqmc.types.Params]): the parameters of the wave function
                ansatz(es), one entry per electronic state.
            rng (~deepqmc.types.KeyArray): an RNG key for stochastic loss components.
            batch (~deepqmc.types.Batch): a batch of physical configurations, importance
                weights, and optional auxiliary data.

        Returns:
            tuple[~jax.Array, tuple[~deepqmc.types.Energy | None,
            ~jax.Array | None, ~deepqmc.types.Stats]]: a scalar loss value and a
            tuple of auxiliary data containing the per-sample local energies,
            optional wave function ratios, and a statistics dictionary.
        """
        ...


class LossFunctionFactory(Protocol):
    r"""Protocol for loss function factories.

    A :class:`LossFunctionFactory` constructs a :class:`LossFunction` from a
    Hamiltonian and an ansatz, encapsulating the choice of objective (energy,
    overlap, spin, …) and any associated hyperparameters.
    """

    def __call__(
        self,
        hamil: MolecularHamiltonian,
        ansatz: Ansatz,
    ) -> LossFunction:
        r"""Construct a loss function for the given system.

        Args:
            hamil (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the
                physical system.
            ansatz (~deepqmc.types.Ansatz): the wave function ansatz.

        Returns:
            :class:`LossFunction`: the loss function for the given Hamiltonian and
            ansatz.
        """
        ...


class LossAndGradFunction(Protocol):
    r"""Protocol for combined loss-and-gradient functions.

    A :class:`LossAndGradFunction` has the same call signature as a
    :class:`LossFunction` but additionally returns the gradient of the loss with
    respect to the model parameters. It is typically obtained by applying
    :func:`jax.value_and_grad` to a :class:`LossFunction`.
    """

    def __call__(
        self,
        params: list[Params],
        rng: KeyArray,
        batch: Batch,
    ) -> tuple[
        tuple[jax.Array, tuple[Energy, Optional[jax.Array], Stats]],
        tuple[jax.Array, tuple[Energy, Optional[jax.Array], Stats]],
    ]:
        r"""Evaluate the loss function and compute its gradient.

        Args:
            params (list[~deepqmc.types.Params]): the parameters of the wave function
                ansatz(es), one entry per electronic state.
            rng (~deepqmc.types.KeyArray): an RNG key for stochastic loss components.
            batch (~deepqmc.types.Batch): a batch of physical configurations, importance
                weights, and optional auxiliary data.

        Returns:
            tuple[tuple[~jax.Array, tuple[~deepqmc.types.Energy,
            ~jax.Array | None, ~deepqmc.types.Stats]], tuple[~jax.Array,
            tuple[~deepqmc.types.Energy, ~jax.Array | None,
            ~deepqmc.types.Stats]]]: a ``(value, gradient)`` pair where both
            elements share the structure ``(loss, (local_energies, wf_ratios,
            stats))``.
        """
        ...
