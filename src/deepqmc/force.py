from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from .ecp.gaussian_type_ecp import GaussianTypeECP
from .hamil import MolecularHamiltonian
from .physics import nuclear_energy
from .sampling.sampling_utils import diffs_to_nearest_nuc
from .types import (
    Energy,
    KeyArray,
    ParametrizedWaveFunction,
    Params,
    PhysicalConfiguration,
    Psi,
)


@jax.grad
def nuclear_force(
    R: jax.Array, phys_conf: PhysicalConfiguration, nuclear_charges: jax.Array
):
    r"""Compute the force resulting from nuclear repulsion.

    Note the negative sign in the return value which turns the gradient into force.
    """
    return -nuclear_energy(jdc.replace(phys_conf, R=R), nuclear_charges)


def make_grad_nuc_wf(wf: ParametrizedWaveFunction, i=None, j=None):
    """Constructs the grad of the wf wrt. nuclei."""

    i = slice(None) if i is None else i
    j = slice(None) if j is None else j

    def grad_nuc_wf(params: Params, phys_conf: PhysicalConfiguration):
        def _wf(R: jax.Array):
            psi = wf(params, jdc.replace(phys_conf, R=R))
            return psi.sign * jnp.exp(psi.log)

        grad_psi = jax.grad(_wf)(phys_conf.R)[i, j]
        return Psi(jnp.sign(grad_psi), jnp.log(jnp.abs(grad_psi)))

    return grad_nuc_wf


def make_grad_nuc_log_wf(wf: ParametrizedWaveFunction):
    """Constructs the grad of the log of the wf wrt. nuclei."""

    def grad_nuc_log_wf(params: Params, phys_conf: PhysicalConfiguration) -> jax.Array:
        def _wf(R, phys_conf):
            return wf(params, jdc.replace(phys_conf, R=R)).log

        return jax.grad(_wf, allow_int=True)(phys_conf.R, phys_conf)

    return grad_nuc_log_wf


def make_grad_log_wf(wf: ParametrizedWaveFunction):
    """Constructs the grad of the log of the wf wrt. electrons."""

    def grad_log_wf(params: Params, phys_conf: PhysicalConfiguration) -> jax.Array:
        def _wf(r, phys_conf):
            return wf(params, jdc.replace(phys_conf, r=r)).log

        return jax.grad(_wf)(phys_conf.r, phys_conf)

    return grad_log_wf


def Q(r: jax.Array, R: jax.Array, c: jax.Array) -> jax.Array:
    """Constructs the Q function of [10.1063/1.1621615]."""
    dists = r[None] - R[:, None]
    force = c[:, None, None] * dists / jnp.linalg.norm(dists, axis=-1, keepdims=True)
    return force.sum(-2)


# Currently not used but could help redeuce fluctuations
# of the bare pp estiamtor
def vnl_ongradpsi(hamil: MolecularHamiltonian, wf: ParametrizedWaveFunction):
    """Constructs the non local potential acting on the wf gradient."""
    n_nuc = len(hamil.mol.coords)

    def vnl_ongradpsi_(
        rng: KeyArray, phys_conf: PhysicalConfiguration, params: Params
    ) -> jax.Array:
        def vnl_ongradpsi_ij(i: int, val: jax.Array) -> jax.Array:
            wfgrad_ij = partial(make_grad_nuc_wf(wf, i // 3, i % 3), params)
            v_nl_ongrad_ij = hamil.potential.nonloc_potential(rng, phys_conf, wfgrad_ij)
            return val.at[i // 3, i % 3].set(v_nl_ongrad_ij)

        wf_value = wf(params, phys_conf)
        V_nl_ongradpsi = jax.lax.fori_loop(
            0, n_nuc * 3, vnl_ongradpsi_ij, jnp.zeros_like(phys_conf.R)
        )
        grad_nuc_psi = make_grad_nuc_wf(wf)(params, phys_conf)
        wf_ratio = (
            jnp.exp(grad_nuc_psi.log - wf_value.log[None, None])
            * wf_value.sign[None, None]
            * grad_nuc_psi.sign
        )
        return V_nl_ongradpsi * wf_ratio

    return vnl_ongradpsi_


def antithetic_sampler(
    phys_conf: PhysicalConfiguration, r_cut: float
) -> tuple[PhysicalConfiguration, PhysicalConfiguration]:
    """Mirrors electons within a cutoff on the closest nuclei."""
    r_nn, _ = diffs_to_nearest_nuc(phys_conf.r, phys_conf.R)
    r_ = phys_conf.r - 2 * r_nn[..., :3] * (r_nn[..., -1] < r_cut**2)[..., None]
    return phys_conf, jdc.replace(phys_conf, r=r_)


def antithetic_wrapper(
    evaluate_force: Callable[[KeyArray, Params, PhysicalConfiguration], jax.Array],
    wf: ParametrizedWaveFunction,
    r_cut: float,
):
    """Wraps force evaluation with an antithetic sampler."""

    def evaluate_force_antithetic(
        rng: KeyArray, params: Params, phys_conf: PhysicalConfiguration
    ):
        # expects estimators that do not require access to the local energies for now.
        phys_conf, phys_conf_ = antithetic_sampler(phys_conf, r_cut)
        log_weight_ = 2 * (wf(params, phys_conf_).log - wf(params, phys_conf).log)
        log_weight = jnp.zeros_like(log_weight_)
        weight_stack = jax.nn.softmax(jnp.stack((log_weight, log_weight_), 0), 0)
        rng, rng_ = jax.random.split(rng)
        force = evaluate_force(rng, params, phys_conf)
        force_ = evaluate_force(rng_, params, phys_conf_)
        force_stack = jnp.stack((force, force_), 0)
        return (weight_stack[:, None, None] * force_stack).sum(0)

    return evaluate_force_antithetic


def evaluate_hf_force_bare(hamil: MolecularHamiltonian, wf: ParametrizedWaveFunction):
    """Constructs bare estimator of the HF force."""
    charges_nuc = hamil.potential.ns_valence

    @jax.grad
    def electronic_force(R, phys_conf):
        return -hamil.potential.local_potential(jdc.replace(phys_conf, R=R))

    def evaluate_hf_force_bare_(
        rng: KeyArray, params: Params, phys_conf: PhysicalConfiguration
    ) -> jax.Array:
        force_nuc = nuclear_force(phys_conf.R, phys_conf, charges_nuc)
        force_elec = electronic_force(phys_conf.R, phys_conf)

        if isinstance(hamil.potential, GaussianTypeECP):
            force_elec -= hamil.potential.grad_nonloc_potential(
                wf, rng, phys_conf, params
            )

        return force_nuc + force_elec

    return evaluate_hf_force_bare_


def evaluate_hf_force_ac_zv(
    hamil: MolecularHamiltonian,
    wf: ParametrizedWaveFunction,
):
    """Constructs ac_zv estimator [10.1063/5.0052266] of the HF force."""
    n_nuc = len(hamil.mol.coords)

    def evaluate_hf_force_ac_zv_(
        rng: KeyArray, params: Params, phys_conf: PhysicalConfiguration
    ) -> jax.Array:

        f_bare = evaluate_hf_force_bare(hamil, wf)(rng, params, phys_conf)
        grad_log_psi = make_grad_nuc_log_wf(wf)(params, phys_conf)
        e_loc, _ = hamil.local_energy(wf)(rng, params, phys_conf)

        def local_energy_grad_wf_i(i: int, val: jax.Array) -> jax.Array:
            eloc_ij, _ = hamil.local_energy(make_grad_nuc_wf(wf, i // 3, i % 3))(
                rng, params, phys_conf
            )
            return val.at[i // 3, i % 3].set(eloc_ij)

        val = jnp.zeros_like(phys_conf.R)
        e_loc_grad_psi = jax.lax.fori_loop(0, n_nuc * 3, local_energy_grad_wf_i, val)
        f_zv = f_bare - ((e_loc_grad_psi - e_loc) * grad_log_psi)
        return f_zv

    return evaluate_hf_force_ac_zv_


def evaluate_hf_force_ac_zvzb(
    hamil: MolecularHamiltonian, wf: ParametrizedWaveFunction
):
    """Constructs ac_zvzb estimator [10.1063/5.0052266] of the HF force."""

    def evaluate_hf_force_ac_zvzb_(
        rng: KeyArray,
        params: Params,
        phys_conf: PhysicalConfiguration,
        e_loc: Energy,
        energy: Energy,
    ) -> jax.Array:
        f_zv = evaluate_hf_force_ac_zv(hamil, wf)(rng, params, phys_conf)
        grad_nuc_log_psi = make_grad_nuc_log_wf(wf)(params, phys_conf)
        f_zb = -2 * (e_loc - energy)[None, None] * grad_nuc_log_psi
        return f_zv + f_zb

    return evaluate_hf_force_ac_zvzb_


def evaluate_hf_force_ac_zvq(hamil: MolecularHamiltonian, wf: ParametrizedWaveFunction):
    """Constructs ac_zvQ estimator [10.1063/1.1621615] of the HF force."""

    def evaluate_hf_force_ac_zvq_(
        params: Params, phys_conf: PhysicalConfiguration
    ) -> jax.Array:

        grad_Q = jax.jacfwd(Q)(phys_conf.r, phys_conf.R, hamil.mol.charges)
        grad_nuc_log_psi = make_grad_log_wf(wf)(params, phys_conf)
        force_nuc = nuclear_force(phys_conf.R, phys_conf, hamil.mol.charges)
        f_zv = (grad_nuc_log_psi[None, None] * grad_Q).sum((-1, -2)) + force_nuc
        return f_zv

    return evaluate_hf_force_ac_zvq_


def evaluate_hf_force_ac_zvzbq(
    hamil: MolecularHamiltonian, wf: ParametrizedWaveFunction
):
    """Constructs ac_zvzbQ estimator [10.1063/1.1621615] of the HF force."""

    def evaluate_hf_force_ac_zvzbq_(
        params: Params, phys_conf: PhysicalConfiguration, e_loc: Energy, energy: Energy
    ) -> jax.Array:
        f_zv = evaluate_hf_force_ac_zvq(hamil, wf)(params, phys_conf)
        f_zb = (
            -2
            * (e_loc - energy)[None, None]
            * Q(phys_conf.r, phys_conf.R, hamil.mol.charges)
        )
        return f_zv + f_zb

    return evaluate_hf_force_ac_zvzbq_
