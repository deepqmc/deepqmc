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


def _bare_dihedral(i0, i1, i2, i3, coords):
    r"""A helper function which computes the dihedral angle.

    WARNING: this function has an ill-defined derivative when the dihedral angle is 0,
    therefore it shouldn't be used on its own. Use the function ``dihedral`` instead.
    """
    v1 = difference(i0, i1, coords, normalize=True)
    v2 = difference(i1, i2, coords, normalize=True)
    v3 = difference(i2, i3, coords, normalize=True)
    u1 = jnp.cross(v1, v2)
    u2 = jnp.cross(v2, v3)
    uu = jnp.dot(u1, u1) * jnp.dot(u2, u2)

    a = jnp.dot(u1, u2) / jnp.sqrt(uu)
    a = jnp.clip(a, -1.0, 1.0)
    non_zero_tval = jnp.arccos(a)
    non_zero_tval *= jnp.where(jnp.dot(u1, jnp.cross(u2, v2)) < 0.0, -1, 1)
    return jnp.where(uu != 0.0, non_zero_tval, jnp.array(0.0))


def _dihedral_gradient(i0, i1, i2, i3, coords):
    r"""A helper function which computes a well-defined gradient of the dihedral."""
    l_u = jnp.linalg.norm(difference(i0, i1, coords, normalize=False))
    l_w = jnp.linalg.norm(difference(i1, i2, coords, normalize=False))
    l_v = jnp.linalg.norm(difference(i2, i3, coords, normalize=False))
    u = difference(i1, i0, coords, normalize=True)
    w = difference(i1, i2, coords, normalize=True)
    v = difference(i2, i3, coords, normalize=True)

    first_term = jnp.cross(u, w) / (l_u * (1 - jnp.dot(u, w) ** 2))
    second_term = jnp.cross(v, w) / (l_v * (1 - jnp.dot(v, w) ** 2))
    third_term = jnp.cross(u, w) * (jnp.dot(u, w)) / (l_w * (1 - jnp.dot(u, w) ** 2))
    fourth_term = jnp.cross(v, w) * (-jnp.dot(v, w)) / (l_w * (1 - jnp.dot(v, w) ** 2))

    gradient = (
        jnp.zeros_like(coords)
        .at[i0]
        .add(first_term)
        .at[i1]
        .add(-first_term)
        .at[i2]
        .add(second_term)
        .at[i3]
        .add(-second_term)
        .at[i1]
        .add(third_term - fourth_term)
        .at[i2]
        .add(-third_term + fourth_term)
    )
    return gradient


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3))
def custom_jvp_dihedral(i0, i1, i2, i3, coords):
    r"""Compute the dihedral angle with a custom JVP rule."""
    v1 = difference(i0, i1, coords, normalize=True)
    v2 = difference(i1, i2, coords, normalize=True)
    v3 = difference(i2, i3, coords, normalize=True)
    u1 = jnp.cross(v1, v2)
    u2 = jnp.cross(v2, v3)
    uu = jnp.dot(u1, u1) * jnp.dot(u2, u2)

    a = jnp.dot(u1, u2) / jnp.sqrt(uu)
    a = jnp.clip(a, -1.0, 1.0)
    non_zero_tval = jnp.arccos(a)
    non_zero_tval *= jnp.where(jnp.dot(u1, jnp.cross(u2, v2)) < 0.0, -1, 1)
    return jnp.where(uu != 0.0, non_zero_tval, jnp.array(0.0))


@custom_jvp_dihedral.defjvp
def dihedral_jvp(i0, i1, i2, i3, primals, tangents):
    r"""The jvp rule of the ``custom_jvp_dihedral``."""
    (coords,) = primals
    (coords_tangent,) = tangents
    primals_out = dihedral(i0, i1, i2, i3, coords)
    jacobian = _dihedral_gradient(i0, i1, i2, i3, coords)
    tangents_out = jnp.dot(jacobian.flatten(), coords_tangent.flatten())
    return primals_out, tangents_out


@jax.custom_vjp
def dihedral(i0, i1, i2, i3, coords):
    r"""Compute the dihedral angle between four atoms.

    This is the top-level, user-facing dihedral function. It has a well defined
    backward-mode gradient (as computed e.g. with ``jax.grad``), and a well defined
    second derivative computed with forward-on-backward AD
    (as done e.g. by ``jax.hessian``).
    If other combinations of ``jax`` differentation transformations are used, jax
    might raise and error, or the derivative might be ill-defined around zero.

    Args:
        i0 (int): index of the atom defining, together with ``i1`` and ``i2``, the
            first of the two half-planes.
        i1 (int): index of the second atom, shared by both half-planes.
        i2 (int): index of the third atom, shared by both half-planes.
        i3 (int): index of the atom defining, together with ``i1`` and ``i2``, the
            second of the two half-planes.
        coords (~jax.Array): Cartesian coordinates of the atoms, of shape
            ``(n_atoms, 3)``.

    Returns:
        ~jax.Array: the signed dihedral angle around the ``i1``-``i2`` bond, in
        radians, in the interval :math:`(-\pi, \pi]`.
    """
    return _bare_dihedral(i0, i1, i2, i3, coords)


def dihedral_fwd(i0, i1, i2, i3, coords):
    r"""The forward pass of the VJP rule of the ``dihedral`` function."""
    return custom_jvp_dihedral(i0, i1, i2, i3, coords), (i0, i1, i2, i3, coords)


def dihedral_bwd(cache, gradient):
    r"""The backward pass of the VJP rule of the ``dihedral`` function."""
    i0, i1, i2, i3, coords = cache
    return None, None, None, None, gradient * _dihedral_gradient(i0, i1, i2, i3, coords)


dihedral.defvjp(dihedral_fwd, dihedral_bwd)
