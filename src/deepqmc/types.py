from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any, NamedTuple, Optional, Protocol
from typing_extensions import TypeAlias

import jax
import jax_dataclasses as jdc


class Psi(NamedTuple):
    r"""Represent wave function values.

    The sign and log of the absolute value of the wave function are stored.
    """

    sign: jax.Array
    log: jax.Array


@jdc.pytree_dataclass
class PhysicalConfiguration:
    r"""Represent physical configurations of electrons and nuclei.

    It currently contains the nuclear and electronic coordinates, along with
    :data:`mol_idx`, which specifies which nuclear configuration a given sample
    was obtained from.
    """

    R: jax.Array
    r: jax.Array
    mol_idx: jax.Array

    def __getitem__(self, idx):
        return self.__class__(
            self.R.__getitem__(idx),
            self.r.__getitem__(idx),
            self.mol_idx.__getitem__(idx),
        )

    def __len__(self):
        return len(self.r)

    @property
    def batch_shape(self):
        assert self.r.shape[:-2] == self.R.shape[:-2] == self.mol_idx.shape
        return self.r.shape[:-2]


Params: TypeAlias = MutableMapping
Stats: TypeAlias = dict
Weight: TypeAlias = jax.Array
Energy: TypeAlias = jax.Array
KeyArray: TypeAlias = jax.Array
SamplerState: TypeAlias = dict
OptState: TypeAlias = Any
DataDict: TypeAlias = dict
Batch: TypeAlias = tuple[PhysicalConfiguration, Weight, Optional[DataDict]]
WaveFunction: TypeAlias = Callable[[PhysicalConfiguration], Psi]
ParametrizedWaveFunction: TypeAlias = Callable[[Params, PhysicalConfiguration], Psi]


class TrainState(NamedTuple):
    r"""Represent the current state of the training."""

    sampler: SamplerState
    params: Params
    opt: OptState


class Ansatz(Protocol):
    r"""Protocol for ansatz objects.

    :class:`~deepqmc.types.Ansatz` objects represent a parametrized wave function
    Ansatz. New types of Ansatzes should implement this protocol to be compatible with
    the DeepQMC software suite. It is assumed that Ansatzes take as input a
    :class:`~deepqmc.types.PhysicalConfiguration` for a single sample of electron and
    nuclei configuration. To handle batches of samples, e.g. during training, the Ansatz
    is ``vmap``-ed automatically by DeepQMC.
    """

    def init(self, rng: KeyArray, phys_conf: PhysicalConfiguration) -> Params:
        r"""Initialize the parameters of the Ansatz.

        Args:
            rng (~deepqmc.types.KeyArray): the RNG key used to generate the initial
                parameters.
            phys_conf (~deepqmc.types.PhysicalConfiguration): a dummy input to the
                network of a single electron and nuclei configuration. The value of
                this can be anything, only its shape information is read.

        Returns:
            ~deepqmc.types.Params: the initial parameters of the Ansatz.
        """
        ...

    def apply(
        self, params: Params, phys_conf: PhysicalConfiguration, return_mos: bool = False
    ) -> Psi:
        r"""Evaluate the Ansatz.

        Args:
            params (~deepqmc.types.Params): the current parameters with which to
                evaluate the Ansatz.
            phys_conf (~deepqmc.types.PhysicalConfiguration): a single sample on which
                to evaluate the Ansatz.
            return_mos (bool): whether to return the many-body orbitals instead of the
                wave function.

        Returns:
            ~deepqmc.types.Psi: the value of the wave function.
        """
        ...
