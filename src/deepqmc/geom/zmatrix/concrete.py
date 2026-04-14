from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional, Type

import jax
import jax.numpy as jnp

from ..general import direction_vector, normed_cross_product, rot_y
from .base import (
    ZMatrix,
    ZMatrixEntry,
    ZMatrixEntryTemplate,
    ZMatrixLine,
    ZMatrixLineTemplate,
    ZMatrixTemplate,
)


class ConcreteZMatrixEntry(ZMatrixEntry):
    """A concrete Z matrix entry with a fixed value."""

    atom_idx: int
    value: jax.Array

    def __init__(self, atom_idx: int, value: jax.Array):
        self.atom_idx = atom_idx
        self.value = value


class ConcreteZMatrixEntryTemplate(ZMatrixEntryTemplate):
    """A template for a concrete Z matrix entry.

    The template includes the atom index of the entry, but it doesn't contain
    information about the value of the entry.
    """

    atom_idx: int
    entry_constructor: Type[ConcreteZMatrixEntry] = ConcreteZMatrixEntry

    def __init__(self, atom_idx: int):
        self.atom_idx = atom_idx

    def concretize(self, value: jax.Array):
        return self.entry_constructor(atom_idx=self.atom_idx, value=value)

    @classmethod
    def from_simplified_config(cls, atom_idx: int | None):
        if atom_idx is None:
            return None
        return cls(atom_idx)


class ConcreteZMatrixLine(ZMatrixLine):
    """A concrete Z matrix line with fixed values for bond, angle, and dihedral."""

    bond: Optional[ConcreteZMatrixEntry]
    angle: Optional[ConcreteZMatrixEntry]
    dihedral: Optional[ConcreteZMatrixEntry]

    @property
    def value(self) -> jax.Array:
        if self.bond is None and self.angle is None and self.dihedral is None:
            return jnp.array([])
        return jnp.stack(
            [
                entry.value
                for entry in [self.bond, self.angle, self.dihedral]
                if entry is not None
            ]
        )


class ConcreteZMatrixLineTemplate(ZMatrixLineTemplate):
    """Template for a concrete Z matrix line."""

    bond: Optional[ConcreteZMatrixEntryTemplate]
    angle: Optional[ConcreteZMatrixEntryTemplate]
    dihedral: Optional[ConcreteZMatrixEntryTemplate]
    line_constructor = ConcreteZMatrixLine

    @classmethod
    def from_simplified_config(
        cls, atom_idxs: Sequence[int | None], charge: int | None = None
    ):
        return cls(
            charge=charge,
            bond=ConcreteZMatrixEntryTemplate.from_simplified_config(atom_idxs[0]),
            angle=ConcreteZMatrixEntryTemplate.from_simplified_config(atom_idxs[1]),
            dihedral=ConcreteZMatrixEntryTemplate.from_simplified_config(atom_idxs[2]),
        )


class ConcreteZMatrix(ZMatrix):
    """Concrete Z matrix representation."""

    lines: Sequence[ConcreteZMatrixLine]

    def __init__(self, lines: Sequence[ConcreteZMatrixLine]):
        self.lines = lines

    def to_cartesian(self) -> jax.Array:
        cartesian = jnp.zeros((0, 3))
        for line in self.lines:
            cartesian = place_next_atom_of_zmatrix(cartesian, line)
        return cartesian

    @property
    def value(self):
        """Return the values of the Z matrix's bond lengths, angles and dihedrals."""
        return jnp.concatenate([line.value for line in self.lines])


class ConcreteZMatrixTemplate(ZMatrixTemplate):
    """Template for a concrete Z matrix representation.

    The template includes the "connectivity" of the Z matrix, i.e. which
    atoms form bonds, angles, and dihedrals, but it doesn't contain information
    about the values of these bond lengths, angles, and dihedrals.
    """

    line_templates: Sequence[ConcreteZMatrixLineTemplate]
    zmatrix_constructor = ConcreteZMatrix
    concretize: Callable[..., ConcreteZMatrix]  # here only for type hinting
    concretize_from_cartesian: Callable[
        ..., ConcreteZMatrix
    ]  # here only for type hinting

    def __init__(self, line_templates: Sequence[ConcreteZMatrixLineTemplate]):
        self.line_templates = line_templates

    @classmethod
    def from_simplified_config(cls, line_templates: Sequence[Any]):
        lines = []
        for line_template in line_templates:
            if not isinstance(line_template, Mapping):
                # Only specifies the atom indices, not the charge
                lines.append(
                    ConcreteZMatrixLineTemplate.from_simplified_config(
                        atom_idxs=line_template
                    )
                )
            else:
                lines.append(
                    ConcreteZMatrixLineTemplate.from_simplified_config(**line_template)
                )
        return cls(lines)


def place_next_atom_of_zmatrix(
    previous_cartesian: jax.Array, zmatrix_line: ConcreteZMatrixLine
) -> jax.Array:
    r"""Place the next atom according to the Z matrix line."""
    if zmatrix_line.bond is None:
        assert len(previous_cartesian) == 0
        return jnp.zeros((1, 3))
    if zmatrix_line.angle is None:
        assert len(previous_cartesian) == 1
        assert zmatrix_line.bond.atom_idx == 0
        return jnp.concatenate(
            [
                previous_cartesian,
                previous_cartesian[zmatrix_line.bond.atom_idx][None]
                + jnp.array([zmatrix_line.bond.value, 0, 0]),
            ],
            axis=0,
        )
    if zmatrix_line.dihedral is None:
        assert len(previous_cartesian) == 2
        assert zmatrix_line.bond.atom_idx < 2
        if zmatrix_line.bond.atom_idx == 0:
            r = jnp.array([zmatrix_line.bond.value, 0, 0])
        else:
            r = -jnp.array([zmatrix_line.bond.value, 0, 0])
        rotated_r = jnp.einsum('ij,j->i', rot_y(zmatrix_line.angle.value), r)
        return jnp.concatenate(
            [
                previous_cartesian,
                rotated_r[None] + previous_cartesian[zmatrix_line.bond.atom_idx],
            ],
            axis=0,
        )

    r_cos_angle = jnp.cos(jnp.pi - zmatrix_line.angle.value) * zmatrix_line.bond.value
    r_sin_angle = jnp.sin(jnp.pi - zmatrix_line.angle.value) * zmatrix_line.bond.value
    bonded_atom_coord = previous_cartesian[zmatrix_line.bond.atom_idx]
    angle_atom_coord = previous_cartesian[zmatrix_line.angle.atom_idx]
    dihedral_atom_coord = previous_cartesian[zmatrix_line.dihedral.atom_idx]

    r = jnp.stack(
        [
            r_cos_angle,
            jnp.cos(zmatrix_line.dihedral.value) * r_sin_angle,
            jnp.sin(zmatrix_line.dihedral.value) * r_sin_angle,
        ]
    )
    BC = direction_vector(bonded_atom_coord, angle_atom_coord)
    AB = direction_vector(angle_atom_coord, dihedral_atom_coord)
    N = normed_cross_product(AB, BC)
    M = normed_cross_product(N, BC)
    rot = jnp.stack([BC, M, N], axis=1)
    r_final = bonded_atom_coord + jnp.dot(rot, r)
    return jnp.concatenate([previous_cartesian, r_final[None]], axis=0)
