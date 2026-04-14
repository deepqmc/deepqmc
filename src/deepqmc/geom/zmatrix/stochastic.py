from collections.abc import Callable, Mapping, Sequence
from typing import Any, Generator, Optional, Protocol, Type

import jax
import jax.numpy as jnp

from ...types import KeyArray
from .base import (
    ZMatrix,
    ZMatrixEntry,
    ZMatrixEntryTemplate,
    ZMatrixLine,
    ZMatrixLineTemplate,
    ZMatrixTemplate,
)
from .concrete import ConcreteZMatrix, ConcreteZMatrixEntry, ConcreteZMatrixLine


def rng_iterator(rng: KeyArray) -> Generator[KeyArray, None, None]:
    """Infinite iterator over random number generators."""
    yield rng
    while True:
        rng, rng_next = jax.random.split(rng)
        yield rng_next


class DistributionFactory(Protocol):
    """Protocol for distribution factories."""

    def __call__(self, loc: jax.Array) -> Callable[[KeyArray], jax.Array]: ...


class StochasticZMatrixEntry(ZMatrixEntry):
    """Stochastic Z matrix entry.

    A stochastic Z matrix entry is made up of an atom index and a function that
    samples a random value for the entry from a given distribution.
    """

    atom_idx: int
    value_fn: Callable[[KeyArray], jax.Array]

    def __init__(self, atom_idx: int, value_fn: Callable[[KeyArray], jax.Array]):
        self.atom_idx = atom_idx
        self.value_fn = value_fn

    def __call__(self, rng: KeyArray) -> ConcreteZMatrixEntry:
        return ConcreteZMatrixEntry(atom_idx=self.atom_idx, value=self.value_fn(rng))


class StochasticZMatrixEntryTemplate(ZMatrixEntryTemplate):
    """Template for a stochastic Z matrix entry."""

    atom_idx: int
    distribution_factory: DistributionFactory
    entry_constructor: Type[StochasticZMatrixEntry] = StochasticZMatrixEntry

    def __init__(
        self,
        atom_idx: int,
        distribution_factory: DistributionFactory,
    ):
        self.atom_idx = atom_idx
        self.distribution_factory = distribution_factory

    def concretize(self, value: jax.Array):
        return self.entry_constructor(
            atom_idx=self.atom_idx, value_fn=self.distribution_factory(value)
        )

    @classmethod
    def from_simplified_config(
        cls,
        atom_idx: int | None,
        distribution_factory: DistributionFactory | None,
    ):
        if atom_idx is None:
            return None
        assert distribution_factory is not None
        return cls(atom_idx=atom_idx, distribution_factory=distribution_factory)


class StochasticZMatrixLine(ZMatrixLine):
    """Stochastic Z matrix line."""

    bond: Optional[StochasticZMatrixEntry]
    angle: Optional[StochasticZMatrixEntry]
    dihedral: Optional[StochasticZMatrixEntry]

    def __call__(self, rng: KeyArray):
        rng_bond, rng_angle, rng_dihedral = jax.random.split(rng, 3)
        bond = None if self.bond is None else self.bond(rng_bond)
        angle = None if self.angle is None else self.angle(rng_angle)
        dihedral = None if self.dihedral is None else self.dihedral(rng_dihedral)
        return ConcreteZMatrixLine(
            self.charge, bond=bond, angle=angle, dihedral=dihedral
        )


class StochasticZMatrixLineTemplate(ZMatrixLineTemplate):
    """Template for a stochastic Z matrix line."""

    bond: Optional[StochasticZMatrixEntryTemplate]
    angle: Optional[StochasticZMatrixEntryTemplate]
    dihedral: Optional[StochasticZMatrixEntryTemplate]
    line_constructor = StochasticZMatrixLine

    @classmethod
    def from_simplified_config(
        cls,
        atom_idxs: Sequence[int | None],
        distribution_factories: Sequence[Callable | None],
        charge: Optional[int] = None,
    ):
        return cls(
            charge,
            bond=StochasticZMatrixEntryTemplate.from_simplified_config(
                atom_idx=atom_idxs[0], distribution_factory=distribution_factories[0]
            ),
            angle=StochasticZMatrixEntryTemplate.from_simplified_config(
                atom_idx=atom_idxs[1], distribution_factory=distribution_factories[1]
            ),
            dihedral=StochasticZMatrixEntryTemplate.from_simplified_config(
                atom_idx=atom_idxs[2], distribution_factory=distribution_factories[2]
            ),
        )


class StochasticZMatrix(ZMatrix):
    """Stochastic Z matrix representation.

    A Z matrix where the values of the bond lengths, angles, and dihedrals are
    defined by a distribution.
    """

    lines: Sequence[StochasticZMatrixLine]

    def __init__(self, lines: Sequence[StochasticZMatrixLine]):
        self.lines = lines

    def __call__(self, rng: KeyArray) -> ConcreteZMatrix:
        lines = [line(rng) for rng, line in zip(rng_iterator(rng), self.lines)]
        return ConcreteZMatrix(lines)

    def to_cartesian(self, rng: KeyArray) -> jax.Array:
        return self(rng).to_cartesian()


class StochasticZMatrixTemplate(ZMatrixTemplate):
    """Template for a stochastic Z matrix.

    The template includes the "connectivity" of the Z matrix, i.e. which
    atoms form bonds, angles, and dihedrals. Moreover, it includes a recipe
    for generating distributions for the bond lengths, angles, and dihedrals.
    """

    line_templates: Sequence[StochasticZMatrixLineTemplate]
    zmatrix_constructor = StochasticZMatrix
    concretize: Callable[..., StochasticZMatrix]  # here only for type hinting
    concretize_from_cartesian: Callable[
        ..., StochasticZMatrix
    ]  # here only for type hinting

    def __init__(self, line_templates: Sequence[StochasticZMatrixLineTemplate]):
        self.line_templates = line_templates

    @classmethod
    def from_simplified_config(cls, lines: Sequence[Mapping[str, Any]]):
        line_templates = [
            StochasticZMatrixLineTemplate.from_simplified_config(**line)
            for line in lines
        ]
        return cls(line_templates)


class UniformDistributionFactory(DistributionFactory):
    """Create uniform distributions."""

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def __call__(self, loc: jax.Array):
        def uniform_distribution(rng: KeyArray):
            return jax.random.uniform(rng, minval=self.low, maxval=self.high)

        return uniform_distribution


class RadiallyUniformDistributionFactory(DistributionFactory):
    """Create radially uniform distributions."""

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def __call__(self, loc: jax.Array):
        def radially_uniform_distribution(rng: KeyArray):
            u = jax.random.uniform(rng)
            r = jnp.sqrt(u * (self.high**2 - self.low**2) + self.low**2)
            return r

        return radially_uniform_distribution


class CenteredRadiallyUniformDistributionFactory(DistributionFactory):
    """Create radially uniform distributions centered on some value."""

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def __call__(self, loc: jax.Array):
        low = loc - self.low
        high = loc + self.high

        def centered_radially_uniform_distribution(rng: KeyArray):
            u = jax.random.uniform(rng)
            r = jnp.sqrt(u * (high**2 - low**2) + low**2)
            return r

        return centered_radially_uniform_distribution


class ClippedAsymmetricNormalDistributionFactory(DistributionFactory):
    """Create clipped, asymmetric normal distributions."""

    def __init__(
        self,
        low_scale: float,
        high_scale: float,
        low: Optional[float] = None,
        high: Optional[float] = None,
    ):
        self.low_scale = low_scale
        self.high_scale = high_scale
        self.low = low
        self.high = high

    def __call__(self, loc: jax.Array):
        def clipped_normal_distribution(rng: KeyArray):
            x = jax.random.normal(rng)
            scale = jnp.where(x > 0, self.high_scale, self.low_scale)
            return jnp.clip(loc + scale * x, self.low, self.high)

        return clipped_normal_distribution


class ClippedNormalDistributionFactory(DistributionFactory):
    """Create clipped normal distributions."""

    def __init__(
        self, scale: float, low: Optional[float] = None, high: Optional[float] = None
    ):
        self.scale = scale
        self.low = low
        self.high = high

    def __call__(self, loc: jax.Array):
        def clipped_normal_distribution(rng: KeyArray) -> jax.Array:
            return jnp.clip(
                loc + self.scale * jax.random.normal(rng), self.low, self.high
            )

        return clipped_normal_distribution


class DeltaDistributionFactory(DistributionFactory):
    """Create delta distributions."""

    def __call__(self, loc: jax.Array):
        def delta_distribution(rng: KeyArray):
            return loc

        return delta_distribution


class CenteredUniformDistributionFactory(DistributionFactory):
    """Create uniform distributions centered on some value."""

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def __call__(self, loc: jax.Array):
        def centered_uniform_distribution(rng: KeyArray):
            return jax.random.uniform(
                rng, minval=loc - self.low, maxval=loc + self.high
            )

        return centered_uniform_distribution
