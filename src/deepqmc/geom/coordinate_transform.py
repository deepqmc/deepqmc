from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar

import jax
import jax.numpy as jnp

from .zmatrix import ConcreteZMatrixTemplate

T = TypeVar('T', covariant=True)  # noqa: N808
C = TypeVar('C')


class CoordinateTransform(Protocol):
    r"""Protocol for coordinate transformations.

    A :class:`CoordinateTransform` maps Cartesian nuclear coordinates to another
    (possibly lower-dimensional) coordinate representation, e.g. a set of internal
    coordinates. It is used e.g. by
    :class:`~deepqmc.sampling.nuclei_samplers.ConstraintNucleiSampler` to perform
    sampling steps in a coordinate system other than Cartesian.
    """

    def from_cartesian(self, coords: jax.Array) -> jax.Array:
        r"""Transform Cartesian nuclear coordinates to this representation.

        Args:
            coords (~jax.Array): Cartesian nuclear coordinates, of shape
                ``(n_nuc, 3)``.

        Returns:
            ~jax.Array: the coordinates in the target representation, of shape
            ``(len(self),)``.
        """
        ...

    def __len__(self) -> int:
        r"""Return the number of coordinates produced by this transform."""
        ...


class InvertibleCoordinateTransform(CoordinateTransform, Protocol):
    r"""Protocol for invertible coordinate transformations.

    In addition to :meth:`~CoordinateTransform.from_cartesian`, an
    :class:`InvertibleCoordinateTransform` can also map coordinates back to
    Cartesian space. This is required to apply e.g. sampled noise in the
    transformed coordinate system, as done by
    :class:`~deepqmc.sampling.nuclei_samplers.ConstraintNucleiSampler`.
    """

    def to_cartesian(self, coords: jax.Array) -> jax.Array:
        r"""Transform coordinates in this representation back to Cartesian space.

        Args:
            coords (~jax.Array): coordinates in this transform's representation,
                of shape ``(len(self),)``.

        Returns:
            ~jax.Array: the corresponding Cartesian nuclear coordinates, of shape
            ``(n_nuc, 3)``.
        """
        ...


class ZMatrixCoordinateTransform(InvertibleCoordinateTransform):
    r"""Invertible coordinate transform between Cartesian coordinates and a Z matrix.

    The transformed coordinates are the flattened bond lengths, angles and dihedral
    angles described by ``zmatrix_template``. This transform is typically passed as
    the ``coordinate_transform`` of a
    :class:`~deepqmc.sampling.nuclei_samplers.ConstraintNucleiSampler`, to sample
    nuclear displacements in terms of bond lengths, angles and dihedrals rather than
    Cartesian coordinates.

    Args:
        zmatrix_template (~deepqmc.geom.zmatrix.ConcreteZMatrixTemplate): the Z
            matrix template defining which atoms are connected by the bonds,
            angles and dihedrals making up the Z matrix.
    """

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
    r"""Identity coordinate transform operating on flattened Cartesian coordinates.

    This is the default ``coordinate_transform`` used by
    :class:`~deepqmc.sampling.nuclei_samplers.ConstraintNucleiSampler` when none is
    specified, i.e. noise is added directly to the Cartesian nuclear coordinates.

    Args:
        n_atoms (int): the number of atoms (nuclei) whose coordinates are
            transformed.
    """

    def __init__(self, n_atoms: int):
        self.n_atoms = n_atoms

    def from_cartesian(self, coords: jax.Array) -> jax.Array:
        return coords.reshape(self.n_atoms * 3)

    def to_cartesian(self, coords: jax.Array) -> jax.Array:
        return coords.reshape(self.n_atoms, 3)

    def __len__(self) -> int:
        return self.n_atoms * 3


class RedundantInternalCoordinateTransform(CoordinateTransform):
    r"""Coordinate transform to a (possibly redundant) set of internal coordinates.

    This transform is not invertible: it merely evaluates a user-specified list of
    internal-coordinate functions, e.g. :func:`~deepqmc.geom.distance`,
    :func:`~deepqmc.geom.angle` or :func:`~deepqmc.geom.dihedral` partially applied
    to fixed atom indices, on the Cartesian nuclear coordinates. Unlike a Z matrix,
    the resulting coordinates need not have a one-to-one correspondence with the
    Cartesian coordinates and may be redundant.

    Args:
        internal_coordinates
            (~collections.abc.Sequence[~collections.abc.Callable[..., ~jax.Array]]):
            a sequence of functions, each called with the Cartesian nuclear
            coordinates as the keyword argument ``coords`` and returning a single
            scalar internal coordinate.
    """

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
    r"""Coordinate transform wrapping an arbitrary user-defined transform function.

    Args:
        n_coordinate (int): the number of coordinates returned by
            ``coordinate_transform_fn``.
        coordinate_transform_fn (~collections.abc.Callable[[~jax.Array], ~jax.Array]):
            a function mapping Cartesian nuclear coordinates to a coordinate array
            of length ``n_coordinate``.
    """

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
    r"""Coordinate transform selecting a subset of the flattened Cartesian coordinates.

    Args:
        coordinate_idxs (~collections.abc.Sequence[int]): indices into the
            flattened (``n_nuc * 3``) Cartesian coordinate array to select.
    """

    def __init__(self, coordinate_idxs):
        self.coordinate_idxs = coordinate_idxs

    def from_cartesian(self, coords: jax.Array) -> jax.Array:
        return coords.flatten()[self.coordinate_idxs]

    def __len__(self) -> int:
        return len(self.coordinate_idxs)
