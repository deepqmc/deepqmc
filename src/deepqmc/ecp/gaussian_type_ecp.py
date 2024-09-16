from typing import Optional

import jax
import jax.numpy as jnp
from pyscf import gto
from scipy.special import legendre

from ..physics import pairwise_distance
from ..types import Energy, KeyArray, PhysicalConfiguration, WaveFunction
from .ecp_utils import (
    get_quadrature_points,
    get_unit_icosahedron_sph,
    pad_list_of_3D_arrays_to_one_array,
)


def parse_gaussian_type_ecp_params(charges, ecp_type, ecp_mask):
    """Load and parse the ECP parameters from the pyscf package.

    This function loads the ECP parameters for an atom (given by `charge`
    argument) from the pyscf package and parses them to jnp arrays.

    Args:
        charges (jax.Array): an array of atomic numbers of the atoms in the molecule
        ecp_type (str): the type of the ECP to load, typically 'bfd' or 'ccECP'
        ecp_mask (jax.Array): an array if booleans indicating whether to use an ECP
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
            pyscf_mole = gto.M(
                atom=[(int(atomic_number), jnp.array([0, 0, 0]))],
                # spin is just a placeholder, ecp parameters don't depend on the spin
                spin=atomic_number % 2,
                ecp=ecp_type,
            )
            assert len(pyscf_mole._ecp) == 1, (
                f'Effective core potential of type {ecp_type} not found for'
                f' {pyscf_mole._atom[0][0]} atom.'
            )
            _, data = pyscf_mole._ecp.popitem()
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


class GaussianTypeECP:
    r"""
    ECPs of the standard semi-local form with the functions given by sums of gaussians.

    Supports ECPs that are defined in pyscf package, such as 'bfd', 'ccECP',
    'ccECP_reg' or 'ccECP_He'. The ECP parameters are loaded directly from
    the pyscf package. The ECP is defined by the general formula:
    :math: `\sum_{l=0}^{l_\text{max}} V_{\text{nl}}(\mathbf{r}) |lm\rangle\langle lm|`
    :math: V_\text{nl}({r}) = \sum_{k=1}^{2} \beta_{lk} \text{e}^{-\alpha_k r^2}
    """

    def __init__(
        self, charges: jax.Array, ecp_type: Optional[str], ecp_mask: jax.Array
    ):
        self.ecp_mask = ecp_mask
        self.ns_valence, self.loc_params, self.nl_params = (
            parse_gaussian_type_ecp_params(charges, ecp_type, ecp_mask)
        )
        # to filter out masked nuclei:
        self.nuc_with_nl_pot = jnp.unique(jnp.nonzero(self.nl_params)[0])
        self.quadrature_thetas = get_unit_icosahedron_sph()[:, 0]

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
        replacing the remaining vmap with fori_loop over the 12 quadrature points.

        Args:
            rng (jax.random.PRNGKey): key used for PRNG.
            phys_conf (:class:`deepqmc.types.PhysicalConfiguration`): electron and
                nuclear coordinates.
            mol (:class:`deepqmc.Molecule`): a molecule that is used to load the
                ECP parameters.
            wf (:class:`~deepqmc.types.WaveFunction): the wave function ansatz.
        """
        assert rng is not None

        # get value of the denominator (which is constant)
        denominator_wf_sign, denominator_wf_exponent = wf(phys_conf)

        def add_nl_potential_for_one_nucleus(i, val):
            nucleus_index = self.nuc_with_nl_pot[i]
            nl_params = self.nl_params[nucleus_index]
            l_max_p1 = nl_params.shape[0]  # l_max_p1 = l_max + 1

            legendre_values = jnp.stack(
                [
                    jnp.polyval(legendre(l).coef, jnp.cos(self.quadrature_thetas))
                    for l in range(l_max_p1)
                ],
                axis=-1,
            )

            quadrature_phys_conf = get_quadrature_points(
                rng, phys_conf.R[nucleus_index], phys_conf
            )

            # (2l+1)/12 coefficient
            coefs = jnp.tile(
                (jnp.arange(l_max_p1) * 2 + 1) / 12, (len(phys_conf), 1)
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
                    denominator_wf_sign
                    * sign
                    * jnp.exp(exponent - denominator_wf_exponent)
                )  # shape (12,)
                wf_ratio_tile = (
                    wf_ratio[..., None] * legendre_values
                )  # shape (12,l_max)
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
                self.ns_valence.astype(int).sum(),
                nl_potential_for_one_nucleus_and_one_electron,
                0.0,
            )
            return val + nl_potential_for_one_nucleus

        total_nl_potential = (
            jax.lax.fori_loop(
                0, len(self.nuc_with_nl_pot), add_nl_potential_for_one_nucleus, 0.0
            )
            if len(self.nuc_with_nl_pot) > 0
            else jnp.array(0.0)
        )

        return total_nl_potential
