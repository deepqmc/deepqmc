from functools import partial

import jax
import jax.numpy as jnp


def difference(i0, i1, coords, normalize=False):
    r"""Compute the difference vector between two atoms."""
    difference = coords[i1] - coords[i0]
    if normalize:
        norm = jnp.linalg.norm(difference)
        return difference / norm
    return difference


def _bare_angle(i0, i1, i2, coords):
    r"""A helper function which computes the bond angle.

    WARNING: this function has an ill-defined derivative when the angle is 0 or pi,
    therefore it shouldn't be used on its own. Use the function ``angle`` instead.
    """
    vi = difference(i1, i0, coords, normalize=True)
    vk = difference(i1, i2, coords, normalize=True)
    a = jnp.clip(jnp.dot(vi, vk), -1, 1)
    return jnp.arccos(a)


def _angle_gradient(i0, i1, i2, coords):
    r"""A helper function which computes a well-defined gradient of the angle."""
    vi = difference(i1, i0, coords, normalize=True)
    vk = difference(i1, i2, coords, normalize=True)
    ww = jnp.cross(vi, vk)
    w = ww / jnp.linalg.norm(ww)
    l_i = jnp.linalg.norm(difference(i1, i0, coords, normalize=False))
    l_k = jnp.linalg.norm(difference(i1, i2, coords, normalize=False))
    first_term = jnp.cross(vi, w) / l_i
    second_term = jnp.cross(w, vk) / l_k
    return (
        jnp.zeros_like(coords)
        .at[i0]
        .set(first_term)
        .at[i2]
        .set(second_term)
        .at[i1]
        .set(-first_term - second_term)
    )


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def custom_jvp_angle(i0, i1, i2, coords):
    r"""Compute the angle with a custom JVP rule."""
    return _bare_angle(i0, i1, i2, coords)


@custom_jvp_angle.defjvp
def angle_jvp(i0, i1, i2, primals, tangents):
    r"""The jvp rule of the ``custom_jvp_angle``."""
    (coords,) = primals
    (coords_tangent,) = tangents
    primals_out = angle(i0, i1, i2, coords)
    jacobian = _angle_gradient(i0, i1, i2, coords)
    tangents_out = jnp.dot(jacobian.flatten(), coords_tangent.flatten())
    return primals_out, tangents_out


@jax.custom_vjp
def angle(i0, i1, i2, coords):
    """Compute the angle between three atoms.

    This is the top-level, user-facing angle function. It has a well defined
    backward-mode gradient (as computed e.g. with ``jax.grad``), and a well defined
    second derivative computed with forward-on-backward AD
    (as done e.g. by ``jax.hessian``).
    If other combinations of ``jax`` differentation transformations are used, jax
    might raise and error, or the derivative might be ill-defined around zero.
    """
    return _bare_angle(i0, i1, i2, coords)


def angle_fwd(i0, i1, i2, coords):
    r"""The forward pass of the VJP rule of the ``angle`` function."""
    return custom_jvp_angle(i0, i1, i2, coords), (i0, i1, i2, coords)


def angle_bwd(cache, gradient):
    r"""The backward pass of the VJP rule of the ``angle`` function."""
    i0, i1, i2, coords = cache
    return None, None, None, gradient * _angle_gradient(i0, i1, i2, coords)


angle.defvjp(angle_fwd, angle_bwd)
