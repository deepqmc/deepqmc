import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from ..types import KeyArray, ParametrizedWaveFunction, Params, PhysicalConfiguration
from .ecp_utils import (
    compute_wf_ratio,
    get_unit_icosahedron_sph,
    single_quadrature_phys_conf,
    sph2cart,
)


def compute_nl_pot_coefs_and_grad_analytical(
    dsit_vec: jax.Array, dists: jax.Array, nl_params: jax.Array
):
    """Compute the coeffs for the nl potentials according to the derivative type."""

    exp_term = jnp.exp(-jnp.einsum('ij,kj->ikj', (dists**2), nl_params[:, 0, :]))
    nl_pot_coefs = jnp.einsum(
        'kj,ikj->ik',
        nl_params[:, 1, :],
        exp_term,
    )

    nl_pot_coefs_grad = -2 * (
        dsit_vec[:, :, None]
        * jnp.einsum(
            'kj,ikj->ikj',
            nl_params[:, 0, :] * nl_params[:, 1, :],
            exp_term,
        )
    ).sum(axis=-1)
    return nl_pot_coefs, nl_pot_coefs_grad


def make_wf_ratio_and_grad(wf: ParametrizedWaveFunction):
    r"""Constructs the function computing the WF ratio and its gradient.

    This version uses a vmapped value_and_grad function to efficiently compute the WF
    ratio and its gradient.
    """
    unit_icosahedron = sph2cart(get_unit_icosahedron_sph())

    def wf_ratio_and_grad_fn(
        params: Params,
        rng: KeyArray,
        nucleus_idx: jax.Array,
        electron_idx: jax.Array,
        phys_conf: PhysicalConfiguration,
    ) -> tuple[jax.Array, jax.Array]:
        def single_wf_ratio_fn(R: jax.Array, unit_quadrature_coordinate: jax.Array):
            pc = jdc.replace(phys_conf, R=R)
            denominator = wf(params, pc)
            quadrature_phys_conf = single_quadrature_phys_conf(
                rng, electron_idx, nucleus_idx, pc, unit_quadrature_coordinate
            )
            numerator = wf(params, quadrature_phys_conf)
            return compute_wf_ratio(numerator, denominator)

        single_wf_ratio_grad_fn = jax.value_and_grad(single_wf_ratio_fn)
        whole_quadrature_wf_ratio_grad_fn = jax.vmap(single_wf_ratio_grad_fn, (None, 0))
        return whole_quadrature_wf_ratio_grad_fn(phys_conf.R, unit_icosahedron)

    return wf_ratio_and_grad_fn
