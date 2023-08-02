import jax
import jax.numpy as jnp
from scipy.special import legendre

from .types import PhysicalConfiguration
from .utils import norm, rot_y, rot_z, sph2cart, triu_flat

__all__ = ()


def pairwise_distance(coords1, coords2):
    return jnp.linalg.norm(coords1[..., :, None, :] - coords2[..., None, :, :], axis=-1)


def pairwise_diffs(coords1, coords2):
    diffs = coords1[..., :, None, :] - coords2[..., None, :, :]
    return jnp.concatenate([diffs, (diffs**2).sum(axis=-1, keepdims=True)], axis=-1)


def pairwise_self_distance(coords, full=False):
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


def nuclear_energy(phys_conf, mol):
    coulombs = triu_flat(
        mol.ns_valence[:, None] * mol.ns_valence
    ) / pairwise_self_distance(phys_conf.R)
    return coulombs.sum()


def electronic_potential(phys_conf):
    dists = pairwise_self_distance(phys_conf.r)
    return (1 / dists).sum(axis=-1)


def local_potential(phys_conf, mol):
    """Return the local or nuclear potential of the whole system.

    Evaluates either the classical nuclear potential V_nuc(r) = -Z/r or the local
    part of the potentials V_loc(r) from [Burkatzki et al. 2007] eq. (4) or
    from [Annaberdiyev et al. 2018] eq. (3). Returns the sum of all the local potential
    contributions from all the electrons and nuclei.
    """

    dists = pairwise_distance(phys_conf.r, phys_conf.R)
    Z_eff = mol.charges - mol.ns_core  # effective charge of the nuclei
    effective_coulomb_potential = -(Z_eff / dists).sum(axis=(-1, -2))
    if not mol.any_pp:
        return effective_coulomb_potential

    loc_params = mol.pp_loc_params
    idxs = mol.pp_mask  # indices of atoms for whom we use pseudopotential

    r_en = dists[:, idxs]  # electron-nucleus distances for all electrons and PP nuclei

    coulomb_term = jnp.einsum(
        'ij,ki->kji', loc_params[idxs, 0, 1, :], 1 / r_en
    ) * jnp.exp(jnp.einsum('ij,ki->kji', -loc_params[idxs, 0, 0, :], r_en**2))
    const_term = jnp.einsum(
        'ij,kji->kji',
        loc_params[idxs, 1, 1, :],
        jnp.exp(jnp.einsum('ij,ki->kji', -loc_params[idxs, 1, 0, :], r_en**2)),
    )
    linear_term = jnp.einsum('ij,ki->kji', loc_params[idxs, 2, 1, :], r_en) * jnp.exp(
        jnp.einsum('ij,ki->kji', -loc_params[idxs, 2, 0, :], r_en**2)
    )

    # Summation is carried over:
    # - individual cores in the molecule (idxs dimension)
    # - individual electrons (1st dimension of 'dists')
    # - potentially over different coeffs for the term of the same type (ccECP case)
    pseudopotential = (coulomb_term + const_term + linear_term).sum(axis=(-1, -2, -3))

    return effective_coulomb_potential + pseudopotential


@jax.jit
def get_unit_icosahedron_sph():
    # Basic definition of unit icosahedron vertices in spherical coordinates.
    unit_icosahedron_sph = []
    unit_icosahedron_sph.append([0, 0])
    unit_icosahedron_sph.append([jnp.pi, 0])
    for j in range(5):
        unit_icosahedron_sph.append([jnp.arctan(2), jnp.pi / 5 * 2 * j])
        unit_icosahedron_sph.append([jnp.pi - jnp.arctan(2), jnp.pi / 5 * (2 * j - 1)])
    return jnp.array(unit_icosahedron_sph)


# UNIT_ICOSAHEDRON is (12,3) array of unit icosahedron vertices
UNIT_ICOSAHEDRON = sph2cart(get_unit_icosahedron_sph())
QUADRATURE_THETAS = get_unit_icosahedron_sph()[:, 0]


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


def nonlocal_potential(rng, phys_conf, mol, wf):
    r"""Calculate the non-local term of the pseudopotential.

    Formulas are based on data from [Burkatzki et al. 2007] or
    [Annaberdiyev et al. 2018]. Numerical calculation of integrals is based
    on [Li et al. 2022] where 12-point icosahedron quadrature is used.
    The current implementation is using jax.lax.fori_loop instead of vmap over
    the index of the rotated electron. This causes roughly 10% slowdown compared
    to plain vmap, but avoids OOM issues. Further OOM errors could be resolved by
    replacing the remaining vmap with fori_loop over the 12 quadrature points.

    Args:
        phys_conf (:class:`deepqmc.types.PhysicalConfiguration`): electron and
            nuclear coordinates.
        mol (:class:`deepqmc.Molecule`): a molecule that is used to load the
            pseudopotential parameters.
        wf (deepqmc.wf.WaveFunction): the wave function ansatz.
    """

    # get value of the denominator (which is constant)
    denominator_wf_sign, denominator_wf_exponent = wf(phys_conf)

    pp_nl_params = jnp.array(mol.pp_nl_params)
    nuc_with_nl_pot = mol.nuc_with_nl_pot  # filter out masked nuclei

    def add_nl_potential_for_one_nucleus(i, val):
        nucleus_index = nuc_with_nl_pot[i]
        nl_params = pp_nl_params[nucleus_index]
        l_max_p1 = nl_params.shape[0]  # l_max_p1 = l_max + 1

        legendre_values = jnp.stack(
            [
                jnp.polyval(legendre(l).coef, jnp.cos(QUADRATURE_THETAS))
                for l in range(l_max_p1)
            ],
            axis=-1,
        )

        quadrature_phys_conf = get_quadrature_points(
            rng, phys_conf.R[nucleus_index], phys_conf
        )

        # (2l+1)/12 coefficient
        coefs = jnp.tile(
            (jnp.arange(l_max_p1) * 2 + 1) / 12, (len(phys_conf.r), 1)
        )  # shape: (N,l_max)

        dists = pairwise_distance(phys_conf.r, phys_conf.R[nucleus_index, None])
        nl_pot_coefs = jnp.einsum(
            'kj,ikj->ikj',
            nl_params[:, 1, :],
            jnp.exp(-jnp.einsum('ij,kj->ikj', (dists**2), nl_params[:, 0, :])),
        ).sum(axis=-1)

        def nl_potential_for_one_nucleus_and_one_electron(
            i,
            val,
            quadrature_phys_conf=quadrature_phys_conf,
            legendre_values=legendre_values,
            coefs=coefs,
            nl_pot_coefs=nl_pot_coefs,
        ):
            # numerator
            sign, exponent = jax.vmap(wf)(quadrature_phys_conf[i])  # shape (12,)
            wf_ratio = (
                denominator_wf_sign * sign * jnp.exp(exponent - denominator_wf_exponent)
            )  # shape (12,)
            wf_ratio_tile = wf_ratio[..., None] * legendre_values  # shape (12,l_max)
            # sum over 12 "virtual" electron configurations
            num_integral_one_e = jnp.sum(wf_ratio_tile, axis=-2)  # shape (1,)
            coef = coefs[i]  # shape (l_max,)
            nl_pot_coef = nl_pot_coefs[i]  # shape (1,)
            nl_potential_one_e = jnp.sum(
                nl_pot_coef * coef * num_integral_one_e, axis=(-1,)
            )  # shape ()
            return val + nl_potential_one_e

        nl_potential_for_one_nucleus = jax.lax.fori_loop(
            0,
            mol.ns_valence.astype(int).sum(),
            nl_potential_for_one_nucleus_and_one_electron,
            0.0,
        )
        return val + nl_potential_for_one_nucleus

    total_nl_potential = jax.lax.fori_loop(
        0, len(nuc_with_nl_pot), add_nl_potential_for_one_nucleus, 0.0
    )

    return total_nl_potential


def laplacian(f):
    def lap(x):
        n_coord = len(x)
        grad_f = jax.grad(f)
        df, grad_f_jvp = jax.linearize(grad_f, x)
        eye = jnp.eye(n_coord)
        d2f = lambda i, val: val + grad_f_jvp(eye[i])[i]
        d2f_sum = jax.lax.fori_loop(0, n_coord, d2f, 0.0)
        return d2f_sum, df

    return lap
