from collections.abc import Callable, Generator, Mapping, Sequence
from typing import Any, Optional, Protocol, Type

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
    r"""Protocol for distribution factories.

    A :class:`DistributionFactory` is called with the value of a bond length, angle
    or dihedral found in a reference geometry, and returns a sampler function for a
    (typically noisy) distribution over that coordinate. This is the extension
    point used to implement custom noise distributions for entries of a
    :class:`StochasticZMatrixTemplate`.
    """

    def __call__(self, loc: jax.Array) -> Callable[[KeyArray], jax.Array]:
        r"""Create a sampler function for a distribution located around ``loc``.

        Args:
            loc (~jax.Array): the value of the coordinate (bond length, angle or
                dihedral) found in the reference geometry, used to center or
                otherwise parametrize the returned distribution.

        Returns:
            ~collections.abc.Callable[[~deepqmc.types.KeyArray], ~jax.Array]: a
            function that samples a value from the distribution, given an rng key.
        """
        ...


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
        r"""Sample a :class:`~deepqmc.geom.zmatrix.ConcreteZMatrix` from the
        distributions of this Z matrix.

        Args:
            rng (~deepqmc.types.KeyArray): an rng key for sampling.

        Returns:
            ~deepqmc.geom.zmatrix.ConcreteZMatrix: a Z matrix with concrete,
            sampled values.
        """
        lines = [line(rng) for rng, line in zip(rng_iterator(rng), self.lines)]
        return ConcreteZMatrix(lines)

    def to_cartesian(self, rng: KeyArray) -> jax.Array:
        r"""Sample this Z matrix and convert it to Cartesian nuclear coordinates.

        Args:
            rng (~deepqmc.types.KeyArray): an rng key for sampling.

        Returns:
            ~jax.Array: the Cartesian nuclear coordinates, of shape ``(n_nuc, 3)``.
        """
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
        r"""Construct a template from a simplified, config-friendly specification.

        This is the constructor typically used from Hydra configs, e.g. to build
        the ``z_matrix_template`` of a Z-matrix-based nuclei sampler
        (``deepqmc.sampling.nuclei_samplers.ZMatrixSampler``).

        Args:
            lines
                (~collections.abc.Sequence[~collections.abc.Mapping[str, typing.Any]]):
                one entry per atom, in the same order as the nuclear charges. Each
                entry is a mapping with the keys:

                - ``atom_idxs``: a sequence ``(bond_atom_idx, angle_atom_idx,
                  dihedral_atom_idx)`` of (up to three) atom indices, using
                  ``None`` for entries that don't apply (e.g. the first three
                  atoms, which don't need a full bond, angle and dihedral).
                - ``distribution_factories``: a sequence of (up to three)
                  :class:`DistributionFactory` instances (or ``None``), one per
                  entry of ``atom_idxs``, used to sample the corresponding bond
                  length, angle or dihedral around the value found in the
                  reference geometry.
                - ``charge`` (optional): the nuclear charge of the atom, only used
                  for bookkeeping.

        Returns:
            StochasticZMatrixTemplate: the resulting Z matrix template.
        """
        line_templates = [
            StochasticZMatrixLineTemplate.from_simplified_config(**line)
            for line in lines
        ]
        return cls(line_templates)


class UniformDistributionFactory(DistributionFactory):
    r"""Create uniform distributions over a fixed, absolute interval.

    Note that the sampled values do not depend on ``loc``, i.e. the reference
    value of the coordinate is ignored.

    Args:
        low (float): the lower bound of the uniform distribution.
        high (float): the upper bound of the uniform distribution.
    """

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def __call__(self, loc: jax.Array):
        def uniform_distribution(rng: KeyArray):
            return jax.random.uniform(rng, minval=self.low, maxval=self.high)

        return uniform_distribution


class RadiallyUniformDistributionFactory(DistributionFactory):
    r"""Create distributions over a fixed, absolute radial interval, with a
    probability density proportional to the sampled value rather than uniform in
    the value itself.

    This samples ``r`` in ``[low, high]`` as the radius of a point picked
    uniformly at random inside an annulus between ``low`` and ``high``, which is
    the appropriate measure e.g. for sampling bond lengths uniformly with respect
    to the enclosed area. The sampled values do not depend on ``loc``.

    Args:
        low (float): the lower bound of the sampled radius.
        high (float): the upper bound of the sampled radius.
    """

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
    r"""Like :class:`RadiallyUniformDistributionFactory`, but centered on ``loc``.

    Samples ``r`` in ``[loc - low, loc + high]``, with a probability density
    proportional to ``r`` rather than uniform in ``r`` itself.

    Args:
        low (float): the offset below ``loc`` of the lower bound of the sampled
            radius.
        high (float): the offset above ``loc`` of the upper bound of the sampled
            radius.
    """

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
    r"""Create normal distributions centered on ``loc``, with different standard
    deviations on either side of ``loc``, clipped to an absolute range.

    Args:
        low_scale (float): the standard deviation used for samples below ``loc``.
        high_scale (float): the standard deviation used for samples above ``loc``.
        low (float | None): optional, an absolute lower bound the samples are
            clipped to.
        high (float | None): optional, an absolute upper bound the samples are
            clipped to.
    """

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
    r"""Create normal distributions centered on ``loc``, clipped to an absolute
    range.

    Args:
        scale (float): the standard deviation of the normal distribution.
        low (float | None): optional, an absolute lower bound the samples are
            clipped to.
        high (float | None): optional, an absolute upper bound the samples are
            clipped to.
    """

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
    r"""Create a degenerate "distribution" that deterministically returns ``loc``.

    Useful to keep a bond length, angle or dihedral fixed at its reference value
    while other entries of the same Z matrix are sampled stochastically.
    """

    def __call__(self, loc: jax.Array):
        def delta_distribution(rng: KeyArray):
            return loc

        return delta_distribution


class CenteredUniformDistributionFactory(DistributionFactory):
    r"""Create uniform distributions centered on ``loc``.

    Args:
        low (float): the offset below ``loc`` of the lower bound of the uniform
            distribution.
        high (float): the offset above ``loc`` of the upper bound of the uniform
            distribution.
    """

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def __call__(self, loc: jax.Array):
        def centered_uniform_distribution(rng: KeyArray):
            return jax.random.uniform(
                rng, minval=loc - self.low, maxval=loc + self.high
            )

        return centered_uniform_distribution
