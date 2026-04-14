import jax
import jax.numpy as jnp

from ..utils import norm


def direction_vector(x: jax.Array, y: jax.Array) -> jax.Array:
    r"""Compute the direction vector from x to y with unit length."""
    difference = x - y
    return difference / jnp.linalg.norm(difference)


def normed_cross_product(x: jax.Array, y: jax.Array) -> jax.Array:
    r"""Computed the normalized cross product between x and y."""
    cross = jnp.cross(x, y)
    return cross / jnp.linalg.norm(cross)


def pairwise_distance(coords1: jax.Array, coords2: jax.Array) -> jax.Array:
    r"""Compute the pairwise distance between two sets of coordinates."""
    return jnp.linalg.norm(coords1[..., :, None, :] - coords2[..., None, :, :], axis=-1)


def pairwise_diffs(coords1: jax.Array, coords2: jax.Array) -> jax.Array:
    r"""Compute the pairwise differences between two sets of coordinates."""
    diffs = coords1[..., :, None, :] - coords2[..., None, :, :]
    return jnp.concatenate([diffs, (diffs**2).sum(axis=-1, keepdims=True)], axis=-1)


def pairwise_self_distance(coords: jax.Array, full: bool = False) -> jax.Array:
    r"""Compute the pairwise self-distance between a set of coordinates."""
    i, j = jnp.triu_indices(coords.shape[-2], k=1)
    diffs = coords[..., :, None, :] - coords[..., None, :, :]
    dists = norm(diffs[..., i, j, :], safe=True, axis=-1)
    if full:
        dists = (
            jnp.zeros(diffs.shape[:-1])
            .at[..., i, j]
            .set(dists)
            .at[..., j, i]
            .set(dists)
        )
    return dists


def rot_y(theta: jax.Array) -> jax.Array:
    """Returns the rotation matrix about y-axis by angle theta."""
    return jnp.array(
        [
            [jnp.cos(theta), jnp.zeros_like(theta), jnp.sin(theta)],
            [jnp.zeros_like(theta), jnp.ones_like(theta), jnp.zeros_like(theta)],
            [-jnp.sin(theta), jnp.zeros_like(theta), jnp.cos(theta)],
        ]
    )


def rot_z(phi: jax.Array) -> jax.Array:
    """Returns the rotation matrix about z-axis by angle phi."""
    return jnp.array(
        [
            [jnp.cos(phi), -jnp.sin(phi), jnp.zeros_like(phi)],
            [jnp.sin(phi), jnp.cos(phi), jnp.zeros_like(phi)],
            [jnp.zeros_like(phi), jnp.zeros_like(phi), jnp.ones_like(phi)],
        ]
    )
