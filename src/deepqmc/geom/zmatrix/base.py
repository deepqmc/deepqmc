from collections.abc import Callable, Sequence
from functools import partial
from itertools import accumulate
from typing import Generic, Optional, Protocol, Self, Type, TypeVar

import jax
import jax.numpy as jnp

from ...geom import angle, dihedral, distance


class ZMatrixEntry(Protocol):
    """Protocol for Z matrix entries."""

    atom_idx: int


E = TypeVar('E', bound=ZMatrixEntry)


class ZMatrixEntryTemplate(Generic[E], Protocol):
    """Protocol for Z matrix entry templates.

    The ``concretize`` method can be used to create a concrete :class:`ZMatrixEntry`
    from the template.
    """

    atom_idx: int
    from_simplified_config: Callable[..., Optional[Self]]
    entry_constructor: Type[E]

    def concretize(self, value: jax.Array) -> E: ...


class ZMatrixLine(Generic[E]):
    """Represents a line in a Z matrix."""

    charge: Optional[int]
    bond: Optional[E]
    angle: Optional[E]
    dihedral: Optional[E]

    def __init__(
        self,
        charge: Optional[int],
        bond: Optional[E],
        angle: Optional[E],
        dihedral: Optional[E],
    ):
        self.charge = charge
        self.bond = bond
        self.angle = angle
        self.dihedral = dihedral


ET = TypeVar('ET', bound=ZMatrixEntryTemplate)
L = TypeVar('L', bound=ZMatrixLine)


class ZMatrixLineTemplate(Generic[ET, L]):
    """Represents a template for a line in a Z matrix."""

    charge: Optional[int]
    bond: Optional[ET]
    angle: Optional[ET]
    dihedral: Optional[ET]
    line_constructor: Type[L]

    def __init__(
        self,
        charge: Optional[int] = None,
        bond: Optional[ET] = None,
        angle: Optional[ET] = None,
        dihedral: Optional[ET] = None,
    ):
        entry_list = [
            entry.atom_idx for entry in [bond, angle, dihedral] if entry is not None
        ]
        assert len(entry_list) == len(set(entry_list))
        self.charge = charge
        self.bond = bond
        self.angle = angle
        self.dihedral = dihedral

    def concretize(self, values: jax.Array) -> L:
        constructor = partial(self.line_constructor, charge=self.charge)  # type: ignore
        if self.bond is None:
            assert len(values) == 0
            return constructor(bond=None, angle=None, dihedral=None)
        assert len(values) > 0
        constructor = partial(constructor, bond=self.bond.concretize(value=values[0]))
        if self.angle is None:
            assert len(values) == 1
            assert self.bond.atom_idx == 0
            return constructor(angle=None, dihedral=None)
        assert len(values) > 1
        constructor = partial(constructor, angle=self.angle.concretize(value=values[1]))
        if self.dihedral is None:
            assert len(values) == 2
            assert self.bond.atom_idx < 2
            assert self.angle.atom_idx < 2
            return constructor(dihedral=None)
        assert len(values) == 3
        return constructor(dihedral=self.dihedral.concretize(value=values[2]))

    def compute_values(self, atom_idx: int, cartesian: jax.Array) -> jax.Array:
        if self.bond is None:
            return jnp.array([])
        if self.angle is None:
            return jnp.array([distance(atom_idx, self.bond.atom_idx, cartesian)])
        if self.dihedral is None:
            return jnp.array(
                [
                    distance(atom_idx, self.bond.atom_idx, cartesian),
                    angle(atom_idx, self.bond.atom_idx, self.angle.atom_idx, cartesian),
                ]
            )
        return jnp.array(
            [
                distance(atom_idx, self.bond.atom_idx, cartesian),
                angle(atom_idx, self.bond.atom_idx, self.angle.atom_idx, cartesian),
                dihedral(
                    atom_idx,
                    self.bond.atom_idx,
                    self.angle.atom_idx,
                    self.dihedral.atom_idx,
                    cartesian,
                ),
            ]
        )


class ZMatrix(Generic[L], Protocol):
    """Protocol for Z matrix representations."""

    lines: Sequence[L]
    to_cartesian: Callable[..., jax.Array]

    def __init__(self, lines: Sequence[L]):
        self.lines = lines

    def __len__(self) -> int:
        return len(self.lines)


LT = TypeVar('LT', bound=ZMatrixLineTemplate)
Z = TypeVar('Z', bound=ZMatrix)


class ZMatrixTemplate(Generic[LT, Z]):
    """Represents a template for a Z matrix."""

    line_templates: Sequence[LT]
    zmatrix_constructor: Type[Z]

    def concretize(self, values: jax.Array) -> Z:
        split_idxs = list(
            accumulate(min(3, i) for i in range(len(self.line_templates) - 1))
        )
        return self.zmatrix_constructor(
            [
                line_template.concretize(line_values)
                for line_values, line_template in zip(
                    jnp.split(values, split_idxs), self.line_templates
                )
            ]
        )

    def concretize_from_cartesian(self, cartesian: jax.Array) -> Z:
        values = jnp.concatenate(
            [
                line_template.compute_values(atom_idx, cartesian)
                for atom_idx, line_template in enumerate(self.line_templates)
            ]
        )
        return self.concretize(values)

    def clean_values(self, values: jax.Array):
        angle_idxs = jnp.cumsum(jnp.array([2, 2] + (len(self) - 2) * [3]))[
            : len(self) - 2
        ]
        dihedral_idxs = jnp.arange(5, 3 * len(self) - 4, 3)
        angle_values = values[angle_idxs]
        dihedral_values = values[dihedral_idxs]

        angle_values = angle_values % (2 * jnp.pi)
        update_angle_value = angle_values > jnp.pi
        angle_values = jnp.where(
            update_angle_value, 2 * jnp.pi - angle_values, angle_values
        )
        dihedral_values = jnp.where(
            update_angle_value[1:], dihedral_values + jnp.pi, dihedral_values
        )
        dihedral_values = ((dihedral_values + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        return (
            values.at[angle_idxs]
            .set(angle_values)
            .at[dihedral_idxs]
            .set(dihedral_values)
        )

    def __len__(self) -> int:
        return len(self.line_templates)
