import jax
import jax.numpy as jnp

from .angle import angle
from .dihedral import dihedral

__all__ = ['distance', 'angle', 'dihedral']


def difference(
    i0: int, i1: int, coords: jax.Array, normalize: bool = False
) -> jax.Array:
    difference = coords[i1] - coords[i0]
    if normalize:
        norm = jnp.linalg.norm(difference)
        return difference / norm
    return difference


def distance(i0: int, i1: int, coords: jax.Array) -> jax.Array:
    r"""Compute the Euclidean distance between two atoms.

    Args:
        i0 (int): index of the first atom.
        i1 (int): index of the second atom.
        coords (~jax.Array): Cartesian coordinates of the atoms, of shape
            ``(n_atoms, 3)``.

    Returns:
        ~jax.Array: the distance between the atoms at indices ``i0`` and ``i1``.
    """
    return jnp.linalg.norm(difference(i0, i1, coords))
