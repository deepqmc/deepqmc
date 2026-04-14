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
    """Compute the distance between two atoms."""
    return jnp.linalg.norm(difference(i0, i1, coords))
