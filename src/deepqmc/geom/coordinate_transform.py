from collections.abc import Callable
from typing import Protocol, Sequence, TypeVar

import jax
import jax.numpy as jnp

from .zmatrix import ConcreteZMatrixTemplate

T = TypeVar('T', covariant=True)  # noqa: N808
C = TypeVar('C')


class CoordinateTransform(Protocol):
    """Protocol for coordinate transformations."""

    def from_cartesian(self, coords: jax.Array) -> jax.Array: ...

    def __len__(self) -> int: ...


class InvertibleCoordinateTransform(CoordinateTransform, Protocol):
    """Protocol for invertible coordinate transformations."""

    def to_cartesian(self, coords: jax.Array) -> jax.Array: ...


class ZMatrixCoordinateTransform(InvertibleCoordinateTransform):
    """Z matrix based invertible coordinate transform."""

    zmatrix_template: ConcreteZMatrixTemplate

    def __init__(self, zmatrix_template: ConcreteZMatrixTemplate):
        self.zmatrix_template = zmatrix_template

    def __len__(self) -> int:
        r"""Return the number of coordinates, not the number of Z matrix lines."""
        return sum(min(i, 3) for i in range(len(self.zmatrix_template)))

    def to_cartesian(self, coords: jax.Array) -> jax.Array:
        coords = self.zmatrix_template.clean_values(coords)
        return self.zmatrix_template.concretize(coords).to_cartesian()

    def from_cartesian(self, coords: jax.Array) -> jax.Array:
        return self.zmatrix_template.concretize_from_cartesian(coords).value


class CartesianCoordinateTransform(InvertibleCoordinateTransform):
    """Cartesian coordinate transform, identity."""

    def __init__(self, n_atoms: int):
        self.n_atoms = n_atoms

    def from_cartesian(self, coords: jax.Array) -> jax.Array:
        return coords.reshape(self.n_atoms * 3)

    def to_cartesian(self, coords: jax.Array) -> jax.Array:
        return coords.reshape(self.n_atoms, 3)

    def __len__(self) -> int:
        return self.n_atoms * 3


class RedundantInternalCoordinateTransform(CoordinateTransform):
    r"""Coordinate transform to redundant internal coordinates."""

    def __init__(self, internal_coordinates: Sequence[Callable[..., jax.Array]]):
        self.internal_coordinates = internal_coordinates

    def from_cartesian(self, coords: jax.Array) -> jax.Array:
        return jnp.array(
            [
                internal_coordinate(coords=coords)
                for internal_coordinate in self.internal_coordinates
            ]
        )

    def __len__(self) -> int:
        return len(self.internal_coordinates)


class GeneralCoordinateTransform(CoordinateTransform):
    """Coordinate transform built from general coordinate functions."""

    def __init__(
        self,
        n_coordinate: int,
        coordinate_transform_fn: Callable[[jax.Array], jax.Array],
    ):
        self.len = n_coordinate
        self.transform = coordinate_transform_fn

    def from_cartesian(self, coords: jax.Array) -> jax.Array:
        return self.transform(coords)

    def __len__(self) -> int:
        return self.len


class SubsetCoordinateTransform(CoordinateTransform):
    """Coordinate transform for a subset of cartesian coordinates."""

    def __init__(self, coordinate_idxs):
        self.coordinate_idxs = coordinate_idxs

    def from_cartesian(self, coords: jax.Array) -> jax.Array:
        return coords.flatten()[self.coordinate_idxs]

    def __len__(self) -> int:
        return len(self.coordinate_idxs)
