from __future__ import annotations

from typing import Optional, Protocol

import jax

from ..hamil import MolecularHamiltonian
from ..types import Ansatz, Batch, Energy, KeyArray, Params, Stats

__all__ = ()


class LossFunction(Protocol):
    def __call__(
        self,
        params: list[Params],
        rng: KeyArray,
        batch: Batch,
    ) -> tuple[jax.Array, tuple[Optional[Energy], Optional[jax.Array], Stats]]: ...


class LossFunctionFactory(Protocol):
    def __call__(
        self,
        hamil: MolecularHamiltonian,
        ansatz: Ansatz,
    ) -> LossFunction: ...


class LossAndGradFunction(Protocol):
    def __call__(
        self,
        params: list[Params],
        rng: KeyArray,
        batch: Batch,
    ) -> tuple[
        tuple[jax.Array, tuple[Energy, Optional[jax.Array], Stats]],
        tuple[jax.Array, tuple[Energy, Optional[jax.Array], Stats]],
    ]: ...
