from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from pyscf.gto.basis import load_ecp
from pyscf.lib.parameters import ELEMENTS

from ..geom.general import pairwise_distance
from ..physics import Potential
from ..types import (
    Energy,
    KeyArray,
    ParametrizedWaveFunction,
    Params,
    PhysicalConfiguration,
    WaveFunction,
)
from .ecp_force_utils import (
    compute_nl_pot_coefs_and_grad_analytical,
    make_wf_ratio_and_grad,
)
from .ecp_utils import (
    compute_wf_ratio,
    get_unit_quadrature,
    pad_list_of_3D_arrays_to_one_array,
    quadrature_legendre_values,
    single_quadrature_phys_conf,
)


def parse_gaussian_type_ecp_params(
    charges: jax.Array, ecp_type: str, ecp_mask: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Load and parse the ECP parameters from the pyscf package.

    This function loads the ECP parameters for an atom (given by `charge`
    argument) from the pyscf package and parses them to jnp arrays.

    Args:
        charges (~jax.Array): an array of atomic numbers of the atoms in the molecule
        ecp_type (str): the type of the ECP to load, typically 'bfd' or 'ccECP'
        ecp_mask (~jax.Array): an array if booleans indicating whether to use an ECP
            for each atom.
    Returns:
        tuple: a tuple containing a an array of integers indicating the numbers of
            valence electron slots, an array of local ECP
            parameters (padded by zeros if each atom has a different shape of local
            parameters), and an array of nonlocal ECP parameters (also
            padded by zeros).
    """

    ns_valence, ecp_loc_params, ecp_nl_params = [], [], []
    max_number_of_same_type_terms = []
    for i, atomic_number in enumerate(charges):
        if ecp_mask[i]:
            data = load_ecp(ecp_type, [ELEMENTS[int(atomic_number)]])
            assert data, (
                f'Effective core potential of type {ecp_type} not found for'
                f' {ELEMENTS[int(atomic_number)]} atom.'
            )
            ecp_loc_param = data[1][0][1][1:4]
            if len(data[1]) > 1:
                ecp_nl_param = jnp.array([di[1][2] for di in data[1][1:]]).swapaxes(
                    -1, -2
                )
            else:
                ecp_nl_param = jnp.array([[[]]])

            max_number_of_same_type_terms.append(len(max(ecp_loc_param, key=len)))
            n_core = data[0]
        else:
            n_core = 0
            ecp_loc_param = [[], [], []]
            ecp_nl_param = jnp.asarray([[[]]])
        ns_valence.append(atomic_number - n_core)
        ecp_loc_params.append(ecp_loc_param)
        ecp_nl_params.append(ecp_nl_param)

    ns_valence = jnp.asarray(ns_valence)

    # We need to pad local parameters with zeros to be able to
    # convert ecp_loc_params from list to jnp.array.
    pad = max(max_number_of_same_type_terms, default=0)
    ecp_loc_param_padded = []
    for ecp_loc_param in ecp_loc_params:
        ecp_loc_param = [pi + [[0, 0]] * (pad - len(pi)) for pi in ecp_loc_param]
        ecp_loc_param_padded.append(jnp.swapaxes(jnp.array(ecp_loc_param), -1, -2))
        # shape (r^n term, coefficient (β) & exponent (α), no. of terms with the same n)
    ecp_loc_params = jnp.array(ecp_loc_param_padded)

    # We also pad the non-local parameters with zeros
    ecp_nl_params = pad_list_of_3D_arrays_to_one_array(ecp_nl_params)

    return ns_valence, ecp_loc_params, jnp.array(ecp_nl_params)


class GaussianTypeECP(Potential):
    r"""
    ECPs of the standard semi-local form with the functions given by sums of gaussians.

    Supports ECPs that are defined in pyscf package, such as 'bfd', 'ccECP',
    'ccECP_reg' or 'ccECP_He'. The ECP parameters are loaded directly from
    the pyscf package. The ECP is defined by the general formula:

    .. math::
        \sum_{l=0}^{l_\text{max}} V_{\text{nl}}(\mathbf{r}) |lm\rangle\langle lm|

    where

    .. math::
        V_\text{nl}(r) = \sum_{k=1}^{2} \beta_{lk} \text{e}^{-\alpha_k r^2}

    Args:
        charges (~jax.Array): the atomic numbers of the nuclei.
        ecp_type (str): the type of the ECP to load from pyscf.
        ecp_mask (~jax.Array): whether to use an ECP for each nucleus.
        ecp_nonlocal_electrons (int | None): optional, if given, only this many
            electrons closest to each ECP nucleus contribute to its non-local term.
            This is an approximation that neglects the contributions of the farther
            electrons, which are very small due to the Gaussian radial functions.
            If :data:`None` (default), all electrons contribute.
        ecp_quadrature (str): the angular quadrature grid of the non-local term,
            ``'by_row'`` (default), ``'icosahedron'``, ``'octahedron'`` or
            ``'tetrahedron'``, see :func:`~deepqmc.ecp.ecp_utils.get_unit_quadrature`.
    """

    def __init__(
        self,
        charges: jax.Array,
        ecp_type: str,
        ecp_mask: jax.Array,
        ecp_nonlocal_electrons: Optional[int] = None,
        ecp_quadrature: str = 'by_row',
    ):
        assert (
            ecp_nonlocal_electrons is None or ecp_nonlocal_electrons >= 1
        ), 'ecp_nonlocal_electrons must be a positive integer or None.'
        self.ecp_mask = ecp_mask
        self.ecp_nonlocal_electrons = ecp_nonlocal_electrons
        self.ecp_quadrature = ecp_quadrature
        self.ns_valence, self.loc_params, self.nl_params = (
            parse_gaussian_type_ecp_params(charges, ecp_type, ecp_mask)
        )
        # to filter out masked nuclei:
        self.nuc_with_nl_pot = jnp.unique(jnp.nonzero(self.nl_params)[0])

        # group the nuclei by their angular quadrature (see get_unit_quadrature), the
        # non-local terms of each group are evaluated in a separate fori_loop
        groups: dict = {}  # n_points: (unit_quadrature, nucleus indices)
        for nucleus_index in self.nuc_with_nl_pot.tolist():
            unit_quadrature = get_unit_quadrature(
                int(charges[nucleus_index]), ecp_quadrature
            )
            groups.setdefault(len(unit_quadrature), (unit_quadrature, []))[1].append(
                nucleus_index
            )
        self.quadrature_groups = [
            (unit_quadrature, jnp.array(nucleus_indices))
            for unit_quadrature, nucleus_indices in groups.values()
        ]

    def local_potential(self, phys_conf: PhysicalConfiguration) -> Energy:
        dists = pairwise_distance(phys_conf.r, phys_conf.R)
        Z_eff = self.ns_valence  # effective charge of the nuclei
        effective_coulomb_potential = -(Z_eff / dists).sum(axis=(-1, -2))
        idxs = self.ecp_mask  # indices of atoms for whom we use ECP

        r_en = dists[:, idxs]  # electron-nucleus distances for all the particles

        coulomb_term = jnp.einsum(
            'ij,ki->kji', self.loc_params[idxs, 0, 1, :], 1 / r_en
        ) * jnp.exp(
            jnp.einsum('ij,ki->kji', -self.loc_params[idxs, 0, 0, :], r_en**2)
        )
        const_term = jnp.einsum(
            'ij,kji->kji',
            self.loc_params[idxs, 1, 1, :],
            jnp.exp(jnp.einsum('ij,ki->kji', -self.loc_params[idxs, 1, 0, :], r_en**2)),
        )
        linear_term = jnp.einsum(
            'ij,ki->kji', self.loc_params[idxs, 2, 1, :], r_en
        ) * jnp.exp(
            jnp.einsum('ij,ki->kji', -self.loc_params[idxs, 2, 0, :], r_en**2)
        )

        # Summation is carried over:
        # - individual cores in the molecule (idxs dimension)
        # - individual electrons (1st dimension of 'dists')
        # - potentially over different coeffs for the term of the same type (ccECP case)
        effective_core_potential = (coulomb_term + const_term + linear_term).sum(
            axis=(-1, -2, -3)
        )

        return effective_coulomb_potential + effective_core_potential

    def closest_electrons(self, dists: jax.Array) -> jax.Array:
        r"""Return the indices of the electrons contributing to a non-local term.

        Args:
            dists (~jax.Array): electron distances from the nucleus, shape
                ``(N_el, 1)``.

        Returns:
            ~jax.Array: all electron indices if :attr:`ecp_nonlocal_electrons` is
            :data:`None`, otherwise the indices of the
            ``min(ecp_nonlocal_electrons, N_el)`` closest electrons, sorted by
            increasing distance.
        """
        if self.ecp_nonlocal_electrons is None:
            return jnp.arange(dists.shape[0])
        n_closest = min(self.ecp_nonlocal_electrons, dists.shape[0])
        return jnp.argsort(dists[:, 0])[:n_closest]

    def nonloc_potential(
        self,
        rng: Optional[KeyArray],
        phys_conf: PhysicalConfiguration,
        wf: WaveFunction,
    ) -> Energy:
        r"""Calculate the non-local term of the ECP.

        Formulas are based on data from [Burkatzki et al. 2007] or
        [Annaberdiyev et al. 2018]. Numerical calculation of integrals is based
        on [Li et al. 2022] where 12-point icosahedron quadrature is used.
        The current implementation is using jax.lax.fori_loop instead of vmap over
        the index of the rotated electron. This causes roughly 10% slowdown compared
        to plain vmap, but avoids OOM issues. Further OOM errors could be resolved by
        replacing the remaining vmap with fori_loop over the quadrature points.
        If :attr:`ecp_nonlocal_electrons` is set, only that many electrons closest to
        each nucleus are included, the contributions of the other electrons are
        neglected.

        Args:
            rng (~deepqmc.types.KeyArray): key used for PRNG.
            phys_conf (~deepqmc.types.PhysicalConfiguration): electron and
                nuclear coordinates.
            wf (~deepqmc.types.WaveFunction): the wave function ansatz.
        """
        assert rng is not None

        # get value of the denominator (which is constant)
        denominator = wf(phys_conf)

        def add_nl_potential_for_one_nucleus(j, val, unit_quadrature, nucleus_indices):
            nucleus_index = nucleus_indices[j]
            nl_params = self.nl_params[nucleus_index]
            l_max_p1 = nl_params.shape[0]  # l_max_p1 = l_max + 1

            # (2l+1)/n_points coefficient
            coefs = jnp.tile(
                (jnp.arange(l_max_p1) * 2 + 1) / len(unit_quadrature),
                (len(phys_conf), 1),
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
                coefs=coefs,
                nl_pot_coefs=nl_pot_coefs,
            ):
                # numerator
                rng_quadrature = jax.random.fold_in(
                    jax.random.fold_in(rng, nucleus_index), i
                )
                quadrature_phys_conf = jax.vmap(
                    single_quadrature_phys_conf, (None, None, None, None, 0)
                )(rng_quadrature, i, nucleus_index, phys_conf, unit_quadrature)
                legendre_values = quadrature_legendre_values(
                    rng_quadrature,
                    phys_conf.r[i],
                    phys_conf.R[nucleus_index],
                    unit_quadrature,
                    l_max_p1,
                )  # shape (n_points, l_max + 1)
                numerator = jax.vmap(wf)(quadrature_phys_conf)  # shape (n_points,)
                wf_ratio = compute_wf_ratio(numerator, denominator)
                wf_ratio_tile = (
                    wf_ratio[..., None] * legendre_values
                )  # shape (n_points,l_max)
                # sum over n_points "virtual" electron configurations
                num_integral_one_e = jnp.sum(wf_ratio_tile, axis=-2)  # shape (1,)
                coef = coefs[i]  # shape (l_max,)
                nl_pot_coef = nl_pot_coefs[i]  # shape (1,)
                nl_potential_one_e = jnp.sum(
                    nl_pot_coef * coef * num_integral_one_e, axis=(-1,)
                )  # shape ()
                return val + nl_potential_one_e

            closest_electrons = self.closest_electrons(dists)
            nl_potential_for_one_nucleus = jax.lax.fori_loop(
                0,
                len(closest_electrons),
                lambda k, val: nl_potential_for_one_nucleus_and_one_electron(
                    closest_electrons[k], val
                ),
                0.0,
            )
            return val + nl_potential_for_one_nucleus

        total_nl_potential = jnp.array(0.0)
        for unit_quadrature, nucleus_indices in self.quadrature_groups:
            total_nl_potential = jax.lax.fori_loop(
                0,
                len(nucleus_indices),
                partial(
                    add_nl_potential_for_one_nucleus,
                    unit_quadrature=unit_quadrature,
                    nucleus_indices=nucleus_indices,
                ),
                total_nl_potential,
            )

        return total_nl_potential

    def grad_nonloc_potential(
        self,
        wf: ParametrizedWaveFunction,
        rng: KeyArray,
        phys_conf: PhysicalConfiguration,
        params: Params,
    ) -> jax.Array:
        def add_nl_potential_for_one_nucleus(j, val, unit_quadrature, nucleus_indices):

            nucleus_index = nucleus_indices[j]
            nl_params = self.nl_params[nucleus_index]
            l_max_p1 = nl_params.shape[0]
            coefs = jnp.tile(
                (jnp.arange(l_max_p1) * 2 + 1) / len(unit_quadrature),
                (len(phys_conf), 1),
            )
            dists = pairwise_distance(phys_conf.r, phys_conf.R[nucleus_index, None])
            dsit_vec = phys_conf.R[nucleus_index, None] - phys_conf.r
            nl_pot_coefs, nl_pot_coefs_grad = compute_nl_pot_coefs_and_grad_analytical(
                dsit_vec, dists, nl_params
            )

            def nl_potential_for_one_nucleus_and_one_electron(
                i,
                val,
                nucleus_index=nucleus_index,
                coefs=coefs,
                nl_pot_coefs=nl_pot_coefs,
                nl_pot_coefs_grad=nl_pot_coefs_grad,
            ):
                # same quadrature rotation as in nonloc_potential for the same rng
                rng_quadrature = jax.random.fold_in(
                    jax.random.fold_in(rng, nucleus_index), i
                )
                wf_ratio, wf_ratio_grad = make_wf_ratio_and_grad(wf, unit_quadrature)(
                    params, rng_quadrature, nucleus_index, i, phys_conf
                )
                # P_l(cos theta_k) depends on the nucleus position via the electron
                # direction, shapes: (n_points, l_max + 1) and (n_points, l_max + 1, 3)
                legendre_fn = partial(
                    quadrature_legendre_values,
                    rng_quadrature,
                    phys_conf.r[i],
                    unit_quadrature=unit_quadrature,
                    l_max_p1=l_max_p1,
                )
                legendre_values = legendre_fn(phys_conf.R[nucleus_index])
                legendre_values_grad = jax.jacfwd(legendre_fn)(
                    phys_conf.R[nucleus_index]
                )
                wf_ratio_tile = wf_ratio[..., None] * legendre_values
                # product rule, shape (n_points, 3, l_max + 1)
                wf_ratio_tile_grad = (
                    wf_ratio_grad[:, nucleus_index, :][..., None]
                    * legendre_values[:, None, :]
                )
                wf_ratio_tile_grad += wf_ratio[:, None, None] * jnp.swapaxes(
                    legendre_values_grad, -1, -2
                )

                # shapes: (l_max + 1,) and (3, l_max + 1)
                num_integral_one_e = jnp.sum(wf_ratio_tile, axis=0)
                num_integral_one_e_wfgrad = jnp.sum(wf_ratio_tile_grad, axis=0)
                coef = coefs[i]  # (l_max + 1,)
                nl_pot_coefs = nl_pot_coefs[i]  # (l_max + 1,)
                nl_pot_coefs_grad = nl_pot_coefs_grad[i]  # (l_max + 1, 3)
                # product rule, summed over the angular momentum channels
                nl_potential_one_e = jnp.sum(
                    (coef * num_integral_one_e)[:, None] * nl_pot_coefs_grad, axis=0
                ) + jnp.sum(nl_pot_coefs * (coef * num_integral_one_e_wfgrad), axis=-1)

                return val + nl_potential_one_e

            closest_electrons = self.closest_electrons(dists)
            nl_potential_for_one_nucleus = jax.lax.fori_loop(
                0,
                len(closest_electrons),
                lambda k, val: nl_potential_for_one_nucleus_and_one_electron(
                    closest_electrons[k], val
                ),
                jnp.zeros(3),
            )
            return val.at[nucleus_index].set(nl_potential_for_one_nucleus)

        grad_nl_potential = jnp.zeros_like(phys_conf.R)
        for unit_quadrature, nucleus_indices in self.quadrature_groups:
            grad_nl_potential = jax.lax.fori_loop(
                0,
                len(nucleus_indices),
                partial(
                    add_nl_potential_for_one_nucleus,
                    unit_quadrature=unit_quadrature,
                    nucleus_indices=nucleus_indices,
                ),
                grad_nl_potential,
            )

        return grad_nl_potential
