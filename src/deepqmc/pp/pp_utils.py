import math

import jax
import jax.numpy as jnp

from ..types import PhysicalConfiguration


@jax.vmap
def sph2cart(sph, r=1):
    """This function transforms from spherical to cartesian coordinates."""
    theta = sph[0]
    phi = sph[1]
    rsin_theta = r * jnp.sin(theta)
    x = rsin_theta * jnp.cos(phi)
    y = rsin_theta * jnp.sin(phi)
    z = r * jnp.cos(theta)
    return jnp.array([x, y, z])


def rot_y(theta):
    """Returns the rotation matrix about y-axis by angle theta."""
    return jnp.array(
        [
            [jnp.cos(theta), jnp.zeros_like(theta), jnp.sin(theta)],
            [jnp.zeros_like(theta), jnp.ones_like(theta), jnp.zeros_like(theta)],
            [-jnp.sin(theta), jnp.zeros_like(theta), jnp.cos(theta)],
        ]
    )


def rot_z(phi):
    """Returns the rotation matrix about z-axis by angle phi."""
    return jnp.array(
        [
            [jnp.cos(phi), -jnp.sin(phi), jnp.zeros_like(phi)],
            [jnp.sin(phi), jnp.cos(phi), jnp.zeros_like(phi)],
            [jnp.zeros_like(phi), jnp.zeros_like(phi), jnp.ones_like(phi)],
        ]
    )


def get_unit_icosahedron_sph():
    """Basic definition of unit icosahedron vertices in spherical coordinates."""
    unit_icosahedron_sph = []
    unit_icosahedron_sph.append([0, 0])
    unit_icosahedron_sph.append([math.pi, 0])
    for j in range(5):
        unit_icosahedron_sph.append([math.atan(2), math.pi / 5 * 2 * j])
        unit_icosahedron_sph.append([math.pi - math.atan(2), math.pi / 5 * (2 * j - 1)])
    return jnp.array(unit_icosahedron_sph)


# UNIT_ICOSAHEDRON is (12,3) array of unit icosahedron vertices
UNIT_ICOSAHEDRON = sph2cart(get_unit_icosahedron_sph())


def get_quadrature_points(rng, nucleus_position, phys_conf):
    """
    Transform :data:`phys_conf` of size (N,3) into an array quadrature points.

    Return a phys_conf of size (N,12,N,3) that includes all the 12 quadrature point
    configurations (i.e. reference electron position is shifted to another icosahedron
    vertex) corresponding to N different reference electron.
    """

    N = len(phys_conf)
    norm = jnp.linalg.norm(phys_conf.r - nucleus_position, axis=-1)
    theta = jnp.arccos(
        jnp.clip((phys_conf.r - nucleus_position)[..., 2] / norm, a_min=-1.0, a_max=1.0)
    )
    phi = jnp.arctan2(
        (phys_conf.r - nucleus_position)[..., 1],
        (phys_conf.r - nucleus_position)[..., 0],
    )
    phi_random = jax.random.uniform(rng, phi.shape, minval=0, maxval=jnp.pi / 5)

    z_rot_random = jnp.moveaxis(rot_z(phi_random), -1, -3)

    y_rot = jnp.moveaxis(rot_y(theta), -1, -3)  # shape: (3,3,num_e) -> (num_e,3,3)

    z_rot = jnp.moveaxis(rot_z(phi), -1, -3)  # shape: (3,3,num_e) -> (num_e,3,3)

    # auxiliary function applying the rotation and translation to be vmapped
    def transform_coordinates(norm, z_rot, y_rot, z_rot_random, r, nucleus_position):
        return norm * (z_rot @ y_rot @ z_rot_random @ r) + nucleus_position

    # vmapping to include N different rotations corresponding to each electron position
    transform_coordinates = jax.vmap(
        transform_coordinates, in_axes=(-1, -3, -3, -3, None, None)
    )
    # vmapping to be able to transform all 12 icosahedron points at the same time
    transform_coordinates = jax.vmap(
        transform_coordinates, in_axes=(None, None, None, None, -2, None)
    )

    quadrature_points = transform_coordinates(
        norm, z_rot, y_rot, z_rot_random, UNIT_ICOSAHEDRON, nucleus_position
    )  # shape: (12,N,3)
    # we still need to pad the quadrature points with other electron's coordinates
    quadrature_points_copied = jnp.tile(quadrature_points, (N, 1, 1, 1))
    rs_copied = jnp.tile(phys_conf.r, (N, 12, 1, 1))
    criterion = jnp.moveaxis(
        jnp.moveaxis(jnp.tile(jnp.eye(N), (12, 3, 1, 1)), -3, -1), -4, -3
    )
    quadrature_rs = jnp.where(
        criterion, quadrature_points_copied, rs_copied
    )  # shape: (N,12,N,3)
    return PhysicalConfiguration(
        jnp.tile(phys_conf.R[None, None], (N, 12, 1, 1)),
        quadrature_rs,
        jnp.broadcast_to(phys_conf.mol_idx, (N, 12)),
    )


def pad_list_of_3D_arrays_to_one_array(list_of_arrays):
    """Pads a list of 3D arrays by adding zeros and stacks them into a single array."""
    shapes = [jnp.asarray(arr).shape for arr in list_of_arrays]
    target_shape = jnp.max(jnp.array(shapes), axis=0)
    padded_arrays = [
        jnp.pad(
            array,
            [(0, target_shape[i] - array.shape[i]) for i in range(3)],
            mode='constant',
        )
        for array in list_of_arrays
    ]
    return jnp.array(padded_arrays)
