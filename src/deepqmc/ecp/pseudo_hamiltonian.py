import os
from collections.abc import Iterable
from xml.etree import ElementTree
from typing import Callable

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np

import deepqmc

from ..physics import LaplacianFactory, Potential, pairwise_distance
from ..types import Energy, PhysicalConfiguration, WaveFunction

ELEMENTS_WITH_EXISTING_PH = {
    16: 'S',
    24: 'Cr',
    25: 'Mn',
    26: 'Fe',
    27: 'Co',
    28: 'Ni',
    29: 'Cu',
    30: 'Zn',
}


def parse_xml(xml_file):
    """
    Parses the XML file containing the PseudoHamiltonian data.

    This part of the code is adapted from
    https://github.com/bytedance/jaqmc/
    blob/8a9b066c78e0097f6c028849d9d51bf1762e1127/jaqmc/pp/ph/data.py#L46
    Original license: Apache-2.0
    """

    tree = ElementTree.parse(xml_file)
    root = tree.getroot()
    header = root.find('header')
    assert header is not None
    zval = header.attrib.get('zval')
    assert type(zval) is str
    n_valence = float(zval)

    def get_data_arr(index):
        data = [
            float(y)
            for x in root[2][index][0][1].text.split('\n')
            for y in x.strip().split(' ')
            if y != ''
        ]
        return np.array(data)

    s_arr = get_data_arr(0)
    d_arr = get_data_arr(2)

    local_nl = d_arr
    v0_nl = s_arr - local_nl

    # This relation should hold: 2 * v0_nl == 3 * v1_nl, thus no need to compute v1_nl
    # p_arr = get_data_arr(1)
    # v1_nl = p_arr - local_nl

    # We add effective charge offset to the local_nl
    return local_nl + v0_nl + n_valence, -v0_nl / 6, n_valence


def load_PH_functions(
    charges: jax.Array, ecp_mask: jax.Array, ph_file_suffix: str = 'cc'
):
    """Loads the pseudo Hamiltonian functions from reference."""
    ns_valence = []
    PH_functions = {}
    rV_loc = []
    rV_L2 = []
    for i, atomic_number in enumerate(charges):
        atomic_number = int(atomic_number)
        if ecp_mask[i]:
            assert (
                atomic_number in ELEMENTS_WITH_EXISTING_PH
            ), f'Pseudo-Hamiltonian for atomic number {atomic_number} not found \
                (probably does not exist!).'
            atom_name = ELEMENTS_WITH_EXISTING_PH[atomic_number]
            if atom_name not in PH_functions:
                # Load the PH functions from the XML file
                dqmc_dir_name = os.path.dirname(deepqmc.__file__)
                xml_file = (
                    f'{dqmc_dir_name}/ecp/ph_data/{atom_name}.{ph_file_suffix}.xml'
                )
                loc_data, l2_data, n_valence = parse_xml(xml_file)
                rx = jnp.linspace(0, 10.0, 10001)
                PH_functions[atom_name] = {
                    'loc': jax.scipy.interpolate.RegularGridInterpolator(
                        [rx], loc_data, fill_value=0.0
                    ),
                    'L2': jax.scipy.interpolate.RegularGridInterpolator(
                        [rx], l2_data, fill_value=0.0
                    ),
                    'n_valence': n_valence,
                }
            rV_loc.append(PH_functions[atom_name]['loc'])
            rV_L2.append(PH_functions[atom_name]['L2'])
            ns_valence.append(PH_functions[atom_name]['n_valence'])
        else:
            ns_valence.append(atomic_number)
    ns_valence = jnp.asarray(ns_valence)
    # ns_valence, a, v_local, v_L^2
    return ns_valence, None, rV_loc, rV_L2


def compute_differential_operator_using_laplacian(
    laplacian_factory: LaplacianFactory,
    Q: jax.Array,
    wf: WaveFunction,
    phys_conf: PhysicalConfiguration,
) -> tuple[jax.Array, jax.Array]:
    """Computes the second-order differential operator.

    The term is given by
    Σ_{iαβ} A_{αβ}(r_i) ∂^2 ψ(r) / ∂r_{iα} ∂r_{iβ}
    This function uses a transformation trick, were the coordinates are first
    transformed to v = Q^-1 r, where A = QQ^T and the differential operator is thus
    expressed as laplacian in v coordinates. A simple (forward) laplacian
    is then used to compute the second-order term.
    """

    # Compute v = Q^-1 r
    v = jax.scipy.linalg.solve_triangular(Q, phys_conf.r, lower=True)  # (N_el, 3)
    v_flat = v.flatten()  # (N_el * 3,)

    def coordinate_transformed_wave_function(v_flat: jax.Array) -> jax.Array:
        """Wave function in the transformed coordinates."""
        v = v_flat.reshape(-1, 3)  # (N_el, 3) back to original shape
        r = jnp.einsum('nxy, ny -> nx', Q, v)  # (N_el, 3) back to original coordinates
        pc = jdc.replace(phys_conf, r=r.reshape(-1, 3))
        return wf(pc).log

    lapl = laplacian_factory(coordinate_transformed_wave_function)
    result = lapl(v_flat)
    laplacian, jacobian_dense_array = result
    lap_log_psis_like_term = laplacian
    jacobian = jacobian_dense_array.reshape(-1, 3)  # (N_el, 3)

    return lap_log_psis_like_term, jacobian


def apply_functions_to_columns(
    functions: Iterable[Callable[[jax.Array], jax.Array]], x: jax.Array
) -> jax.Array:
    r"""Compute :math:`f^J_(x_{iJ})`.

    Args:
        functions (Iterable[Callable[[jax.Array], jax.Array]]):
            functions to apply, iterable of length ``n_funcs``.
        x (jax.Array): input to the functions. Shape: ``(input_dim, n_funcs)``.
    """
    fx = jnp.stack([fn(xx) for fn, xx in zip(functions, x.T, strict=True)], axis=1)
    return fx


class PseudoHamiltonian(Potential):
    """
    Class for the pseudo Hamiltonian.

    The pseudo Hamiltonian which is fully local unlike the ECP significantly speeding-up
    the computation. The PHs are taken from [Ichibha23] and [Fu25].
    """

    def __init__(self, charges: jax.Array, ecp_type: str, ecp_mask: jax.Array):
        self.ecp_mask = ecp_mask
        self.ns_valence, self.a, self.rV_loc, self.rV_L2 = load_PH_functions(
            charges, ecp_mask
        )

    def local_potential(self, phys_conf: PhysicalConfiguration) -> Energy:
        """Computes the zeroth-order local PseudoHamiltonian term."""
        dists = pairwise_distance(
            phys_conf.r, phys_conf.R
        )  # |r - R|, shape = (N_el, N_nuc)
        Z_eff = self.ns_valence  # effective nuclear charge
        V_Coul_eff = -(Z_eff / dists).sum(
            axis=(-1, -2)
        )  # effective Coulomb repulsion, summed for all nuclei

        dists_PH = dists[:, self.ecp_mask]  # (N_el, N_nuc_ph,)
        rVph_loc = apply_functions_to_columns(self.rV_loc, dists_PH)
        Vph_loc = rVph_loc / dists_PH  # (N_el, N_nuc_ph,)
        Vph_loc = Vph_loc.sum(axis=(-2, -1))  # Sum over all electrons & nuclei

        return V_Coul_eff + Vph_loc

    def compute_coefficients_of_differential_operators(
        self, phys_conf: PhysicalConfiguration
    ) -> tuple[jax.Array, jax.Array]:
        r"""Compute the coefficients of the differential operators.

        Compute the coefficients :math:`A` and :math:`b` in:
        :math:`Σ_{iαβ} A_{αβ}(r_i) ∂^2 ψ(r) / ∂r_{iα} ∂r_{iβ}`
        :math:`+ Σ_{iα} b_{α}(r_i) ∂ ψ(r) / ∂r_{iα}`
        """
        dists = pairwise_distance(phys_conf.r, phys_conf.R)  # shape = (N_el, N_nuc)
        pairwise_diffs = (
            phys_conf.r[..., :, None, :] - phys_conf.R[..., None, :, :]
        )  # shape = (N_el, N_nuc, 3)

        # evaluate the L^2 momentum PH function at |r - R|
        dists_PH = dists[:, self.ecp_mask]  # (N_el, N_nuc_ph,)
        pairwise_diffs_PH = pairwise_diffs[:, self.ecp_mask, :]  # (N_el, N_nuc_ph, 3)
        rVph_L2 = apply_functions_to_columns(self.rV_L2, dists_PH)
        Vph_L2 = rVph_L2 / dists_PH  # (N_el, N_nuc_ph,)

        # compute the PH coefficients for the b vector
        b_I = 2 * Vph_L2[..., None] * pairwise_diffs_PH  # shape = (N_el, N_nuc_ph, 3)
        b = b_I.sum(axis=-2)  # sum over N_nuc_ph nuclei -> shape = (N_el, 3)

        # compute the PH coefficients for the A matrix
        diag_term = rVph_L2 * dists_PH  # shape = (N_el, N_nuc_ph)
        diag_term = diag_term[..., None, None] * jnp.eye(3)
        # diag_term.shape = (N_el, N_nuc_ph, 3, 3)
        nondiag_term = (
            Vph_L2[..., None, None]
            * pairwise_diffs_PH[..., :, None]
            * pairwise_diffs_PH[..., None, :]
        )  # shape = (N_el, N_nuc_ph, 3, 3)
        total_term = (diag_term - nondiag_term).sum(axis=-3)  # shape = (N_el, 3, 3)
        # add 0.5 to the diagonal (for the kinetic term), shape = (N_el, 3, 3)
        A = total_term + 0.5 * jnp.eye(3)
        return A, b

    def kinetic_term(
        self,
        phys_conf: PhysicalConfiguration,
        wf: WaveFunction,
        laplacian_factory: LaplacianFactory,
    ) -> tuple[Energy, jax.Array, jax.Array]:
        """
        Computes the kinetic-like term of the pseudo Hamiltonian.

        That is, all the terms that include first- or second-order differential
        operators. Those terms are
        Σ_{iαβ} A_{αβ}(r_i) ∂^2 ψ(r) / ∂r_{iα} ∂r_{iβ}
        + Σ_{iα} b_{α}(r_i) ∂ ψ(r) / ∂r_{iα}
        where A and b are matrix and vector functions determined by the PH.
        """
        A, b = self.compute_coefficients_of_differential_operators(phys_conf)
        # lower=True must be passed here in order to get A = Q @ Q.T
        # with lower=False, A = Q.T @ Q would be computed, and different
        # transposes would need to be taken elsewhere
        Q = jax.scipy.linalg.cholesky(A, lower=True)

        lap_log_psis_like_term, jacobian_v = (
            compute_differential_operator_using_laplacian(
                laplacian_factory, Q, wf, phys_conf
            )
        )

        # Now we compute the Jacobian in the original coordinates instead of
        # the transformed ones, using the chain rule.
        jacobian_r = jax.scipy.linalg.solve_triangular(
            Q, jacobian_v, trans='T', lower=True
        )
        first_order_term = (b * jacobian_r).sum(axis=(-2, -1))

        quantum_force_like_term = (jacobian_v * jacobian_v).sum(axis=(-2, -1))
        second_order_term = -(
            lap_log_psis_like_term + quantum_force_like_term
        )  # the sign should be correct here

        return (
            first_order_term + second_order_term,
            lap_log_psis_like_term,
            quantum_force_like_term,
        )
