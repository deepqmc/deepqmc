import math

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from scipy.special import legendre

from ..types import Psi


def get_unit_icosahedron():
    """The 12 vertices of a regular icosahedron, cyclic permutations of (0, ±1, ±φ)."""
    phi = (1 + math.sqrt(5)) / 2  # golden ratio
    vertices = jnp.array([[0.0, s1, s2 * phi] for s1 in (-1, 1) for s2 in (-1, 1)])
    vertices = jnp.concatenate([jnp.roll(vertices, k, axis=1) for k in range(3)])
    return vertices / math.sqrt(1 + phi**2)


def get_unit_quadrature(atomic_number, quadrature='by_row'):
    r"""Return the unit quadrature grid.

    Args:
        atomic_number (int): the atomic number of the element.
        quadrature (str): ``'tetrahedron'`` (4 points), ``'octahedron'`` (6 points),
            ``'icosahedron'`` (12 points) or ``'by_row'``.

    Returns:
        the quadrature points on the unit sphere, shape ``(n_points, 3)``.
    """
    if quadrature not in ('by_row', 'icosahedron', 'octahedron', 'tetrahedron'):
        raise ValueError(f'Unknown ECP quadrature {quadrature!r}.')
    if quadrature == 'by_row':
        quadrature = (
            'tetrahedron'
            if atomic_number <= 10
            else 'octahedron' if atomic_number <= 18 else 'icosahedron'
        )
    if quadrature == 'tetrahedron':
        return jnp.array(
            [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
        ) / math.sqrt(3)
    if quadrature == 'octahedron':
        return jnp.concatenate([jnp.eye(3), -jnp.eye(3)])
    else:
        return get_unit_icosahedron()


def random_rotation(rng):
    r"""Sample a rotation matrix uniformly (Haar measure) from SO(3).

    Built from a uniformly distributed unit quaternion.
    """
    q = jax.random.normal(rng, (4,))
    w, x, y, z = q / jnp.linalg.norm(q)
    return jnp.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


def single_quadrature_point(
    rng, electron_coordinate, nucleus_coordinate, unit_quadrature_coordinate
):
    r"""Compute a single quadrature point (electron position).

    The complete quadrature is a randomly rotated unit quadrature centered on
    ``nucleus_coordinate``, with radius given by the distance of
    ``electron_coordinate`` and ``nucleus_coordinate``. The rotation is drawn
    uniformly from SO(3), independently of the electron position.

    Args:
        rng: random seed, determines the rotation of the quadrature.
        electron_coordinate: the coordinates  of the electron.
        nucleus_coordinate: the coordinates of the nucleus.
        unit_quadrature_coordinate: the coordinates of one point of the unit
            quadrature, determines which quadrature point to compute.
    """
    radius = jnp.linalg.norm(electron_coordinate - nucleus_coordinate, axis=-1)
    return (
        radius * (random_rotation(rng) @ unit_quadrature_coordinate)
        + nucleus_coordinate
    )


def quadrature_legendre_values(
    rng, electron_coordinate, nucleus_coordinate, unit_quadrature, l_max_p1
):
    r"""Evaluate the Legendre polynomials at the quadrature points.

    Computes :math:`P_l(\cos\theta_k)`, where :math:`\theta_k` is the angle between
    the electron and the :math:`k`-th (rotated) quadrature point, as seen from the
    nucleus.

    Args:
        rng: random seed, must be the same as in :func:`single_quadrature_point`.
        electron_coordinate: the coordinates  of the electron.
        nucleus_coordinate: the coordinates of the nucleus.
        unit_quadrature: the points of the unit quadrature, shape ``(n_points, 3)``.
        l_max_p1 (int): the number of angular momentum channels.

    Returns:
        the Legendre values, shape ``(n_points, l_max_p1)``.
    """
    diff_vector = electron_coordinate - nucleus_coordinate
    cos_theta = (unit_quadrature @ random_rotation(rng).T) @ (
        diff_vector / jnp.linalg.norm(diff_vector, axis=-1)
    )
    return jnp.stack(
        [jnp.polyval(legendre(l).coef, cos_theta) for l in range(l_max_p1)], axis=-1
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
