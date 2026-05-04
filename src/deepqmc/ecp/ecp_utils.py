import math

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from ..geom.general import rot_y, rot_z
from ..types import Psi


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


def get_unit_icosahedron_sph():
    """Basic definition of unit icosahedron vertices in spherical coordinates."""
    unit_icosahedron_sph = []
    unit_icosahedron_sph.append([0, 0])
    unit_icosahedron_sph.append([math.pi, 0])
    for j in range(5):
        unit_icosahedron_sph.append([math.atan(2), math.pi / 5 * 2 * j])
        unit_icosahedron_sph.append([math.pi - math.atan(2), math.pi / 5 * (2 * j - 1)])
    return jnp.array(unit_icosahedron_sph)


def single_quadrature_point(
    rng, electron_coordinate, nucleus_coordinate, unit_quadrature_coordinate
):
    r"""Compute a single quadrature point (electron position).

    The complete quadrature is a rotated icosahedron centered on
    ``nucleus_coordinate``, with radius and orientation determined by the relative
    positions of ``electron_coordinate`` and ``nucleus_coordinate``. A random rotation
    around the Z axis is also applied to the icosahedron.

    Args:
        rng: random seed.
        electron_coordinate: the coordinates  of the electron.
        nucleus_coordinate: the coordinates of the nucleus.
        unit_quadrature_coordinate: the coordinates of one point of the unit
            icosahedron, determines which quadrature point to compute.
    """
    diff_vector = electron_coordinate - nucleus_coordinate
    radius = jnp.linalg.norm(diff_vector, axis=-1)
    theta = jnp.arccos(jnp.clip(diff_vector[2] / radius, a_min=-1.0, a_max=1.0))
    phi = jnp.arctan2(diff_vector[1], diff_vector[0])
    phi_random = jax.random.uniform(rng, (), minval=0, maxval=jnp.pi / 5)
    return (
        radius
        * (rot_z(phi) @ rot_y(theta) @ rot_z(phi_random) @ unit_quadrature_coordinate)
        + nucleus_coordinate
    )


def single_quadrature_phys_conf(
    rng, electron_idx, nucleus_idx, phys_conf, unit_quadrature_coordinate
):
    r"""Compute a quadrature point as :class:`~deepqmc.types.PhysicalConfiguration`."""
    quadrature_coordinate = single_quadrature_point(
        rng,
        phys_conf.r[electron_idx],
        phys_conf.R[nucleus_idx],
        unit_quadrature_coordinate,
    )
    return jdc.replace(
        phys_conf, r=phys_conf.r.at[electron_idx].set(quadrature_coordinate)
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


def compute_wf_ratio(numerator: Psi, denonimator: Psi) -> jax.Array:
    """Computes the ratio of two wavefunctions."""
    return jnp.exp(numerator.log - denonimator.log) * numerator.sign * denonimator.sign
