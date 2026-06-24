from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any, NamedTuple, Optional, Protocol

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


type Params = MutableMapping
"""Alias for :class:`~collections.abc.MutableMapping`. A nested dictionary like object holding the parameters of a haiku neural network ansatz."""
type Stats = dict
"""Alias for :class:`dict`. A dictionary that is used to gather data for logging."""
type Weight = jax.Array
"""Alias for :class:`~jax.Array`. An array holding importance weights of electron configurations used for weighted averages."""
type Energy = jax.Array
"""Alias for :class:`~jax.Array`. An array holding the local energies of a batch of electron configurations."""
type KeyArray = jax.Array
"""Alias for :class:`~jax.Array`. An array holding data to generate random numbers."""
type SamplerState = dict
"""Alias for :class:`dict`. The state dict of any sampler, holding various data needed for MCMC sampling."""
type OptState = Any
"""Alias for :data:`~typing.Any`. The state object of an optimizer, holding various data needed for optimization."""
type DataDict = dict
"""Alias for :class:`dict`. A dictionary holding auxiliary data used actively in the training, i.e. for scaling losses."""
type Batch = tuple[PhysicalConfiguration, Weight, Optional[DataDict]]
r"""Alias for tuple\[:class:`~deepqmc.types.PhysicalConfiguration`, :data:`~deepqmc.types.Weight`, :data:`~deepqmc.types.DataDict` | None\]. A tuple holding a PhysicalConfiguration, importance weight and optionally auxiliary data of a batch."""
type WaveFunction = Callable[[PhysicalConfiguration], Psi]
r"""Alias for :class:`~collections.abc.Callable`\[\[:class:`~deepqmc.types.PhysicalConfiguration`\], :class:`~deepqmc.types.Psi`\]. A wave function that maps a Physical configuration to the (log) value of the wave function."""
type ParametrizedWaveFunction = Callable[[Params, PhysicalConfiguration], Psi]
r"""Alias for :class:`~collections.abc.Callable`\[\[:data:`~deepqmc.types.Params`, :class:`~deepqmc.types.PhysicalConfiguration`\], :class:`~deepqmc.types.Psi`\]. A wave function that requires model parameters to be provided for its evaluation."""
type OptimizerFactory = Callable[[LossAndGradFunction], Optimizer]  # pyright: ignore
r"""Alias for :class:`~collections.abc.Callable`\[\[:class:`~deepqmc.loss.LossAndGradFunction`\], :class:`~deepqmc.optimizer.Optimizer`\]. A factory function that returns an Optimizer instance from a loss (and gradient) function."""
type SamplerFactory = Callable[
    [
        KeyArray,
        MolecularHamiltonian,  # pyright: ignore
        Ansatz,
        list[Molecule],  # pyright: ignore
        int,
        int,
    ],  # pyright: ignore
    tuple[MoleculeIdxSampler, MultiNuclearGeometrySampler],  # pyright: ignore
]
r"""Alias for :class:`~collections.abc.Callable`\[\[:data:`~deepqmc.types.KeyArray`, :class:`~deepqmc.hamil.MolecularHamiltonian`, :class:`~deepqmc.types.Ansatz`, list\[:class:`~deepqmc.molecule.Molecule`\], int, int\], tuple\[:class:`~deepqmc.sampling.combined_samplers.MoleculeIdxSampler`, :class:`~deepqmc.sampling.combined_samplers.MultiNuclearGeometrySampler`\]\]. A factory function that returns a tuple of a molecule index sampler and an electron and nuclei sampler."""
type AnsatzFactory = Callable[[MolecularHamiltonian], Ansatz]  # pyright: ignore
r"""Alias for :class:`~collections.abc.Callable`\[\[:class:`~deepqmc.hamil.MolecularHamiltonian`\], :class:`~deepqmc.types.Ansatz`\]. A factory function that returns a haiku object that can be transformed to obtain a wave function ansatz."""


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
    is ``vmap``-ed automatically by DeepQMC. The apply function of the Ansatz object is
    a :func:`~deepqmc.types.ParametrizedWaveFunction`.
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
