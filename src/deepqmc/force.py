from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from .hamil import MolecularHamiltonian
from .physics import coulomb_force
from .types import (
    Energy,
    ParametrizedWaveFunction,
    Params,
    PhysicalConfiguration,
    Psi,
)


def make_grad_nuc_wf(
    wf: ParametrizedWaveFunction, i=None, j=None
) -> Callable[[Params, PhysicalConfiguration], Psi]:
    """Constructs the grad of the wf wrt. nuclei."""

    i = slice(None) if i is None else i
    j = slice(None) if j is None else j

    def grad_nuc_wf(params: Params, phys_conf: PhysicalConfiguration) -> Psi:
        def _wf(R):
            psi = wf(params, jdc.replace(phys_conf, R=R))
            return psi.sign * jnp.exp(psi.log)

        grad_psi = jax.grad(_wf)(phys_conf.R)[i, j]
        return Psi(jnp.sign(grad_psi), jnp.log(jnp.abs(grad_psi)))

    return grad_nuc_wf


def make_grad_nuc_log_wf(
    wf: ParametrizedWaveFunction,
) -> Callable[[Params, PhysicalConfiguration], jax.Array]:
    """Constructs the grad of the log of the wf wrt. nuclei."""

    def grad_nuc_log_wf(params: Params, phys_conf: PhysicalConfiguration) -> jax.Array:
        def _wf(R, phys_conf):
            return wf(params, jdc.replace(phys_conf, R=R)).log

        return jax.grad(_wf, allow_int=True)(phys_conf.R, phys_conf)

    return grad_nuc_log_wf


def make_grad_log_wf(
    wf: ParametrizedWaveFunction,
) -> Callable[[Params, PhysicalConfiguration], jax.Array]:
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


def evaluate_hf_force_bare(
    hamil: MolecularHamiltonian,
) -> Callable[[Params, PhysicalConfiguration], jax.Array]:
    """Constructs bare estimator of the HF force."""
    charges_nuc = hamil.mol.charges
    charges_elec = -1 * jnp.ones(hamil.n_up + hamil.n_down)

    def evaluate_hf_force_bare_(
        params: Params, phys_conf: PhysicalConfiguration
    ) -> jax.Array:
        force_nuc = coulomb_force(
            phys_conf.R, phys_conf.R, charges_nuc, charges_nuc, True
        )
        force_elec = coulomb_force(phys_conf.R, phys_conf.r, charges_nuc, charges_elec)
        return force_nuc + force_elec

    return evaluate_hf_force_bare_


def evaluate_hf_force_ac_zv(
    hamil: MolecularHamiltonian, wf: ParametrizedWaveFunction
) -> Callable[[Params, PhysicalConfiguration], jax.Array]:
    """Constructs ac_zv estimator [10.1063/5.0052266] of the HF force."""
    n_nuc = len(hamil.mol.coords)

    evaluate_hf_force_bare_ = evaluate_hf_force_bare(hamil)

    def evaluate_hf_force_ac_zv_(
        params: Params, phys_conf: PhysicalConfiguration
    ) -> jax.Array:

        f_bare = evaluate_hf_force_bare_(params, phys_conf)
        grad_log_psi = make_grad_nuc_log_wf(wf)(params, phys_conf)
        e_loc, _ = hamil.local_energy(wf)(None, params, phys_conf)

        def local_energy_grad_wf_i(i: int, val: jax.Array) -> jax.Array:
            eloc_ij, _ = hamil.local_energy(make_grad_nuc_wf(wf, i // 3, i % 3))(
                None, params, phys_conf
            )  # We can provide None as rng, as forces are not implemented for ecp
            return val.at[i // 3, i % 3].set(eloc_ij)

        val = jnp.zeros_like(phys_conf.R)
        e_loc_grad_psi = jax.lax.fori_loop(0, n_nuc * 3, local_energy_grad_wf_i, val)
        f_zv = f_bare - ((e_loc_grad_psi - e_loc) * grad_log_psi)
        return f_zv

    return evaluate_hf_force_ac_zv_


def evaluate_hf_force_ac_zvzb(
    hamil: MolecularHamiltonian, wf: ParametrizedWaveFunction
) -> Callable[[Params, PhysicalConfiguration, Energy, Energy], jax.Array]:
    """Constructs ac_zvzb estimator [10.1063/5.0052266] of the HF force."""

    def evaluate_hf_force_ac_zvzb_(
        params: Params, phys_conf: PhysicalConfiguration, e_loc: Energy, energy: Energy
    ) -> jax.Array:
        f_zv = evaluate_hf_force_ac_zv(hamil, wf)(params, phys_conf)
        grad_nuc_log_psi = make_grad_nuc_log_wf(wf)(params, phys_conf)
        f_zb = -2 * (e_loc - energy)[None, None] * grad_nuc_log_psi
        return f_zv + f_zb

    return evaluate_hf_force_ac_zvzb_


def evaluate_hf_force_ac_zvq(
    hamil: MolecularHamiltonian, wf: ParametrizedWaveFunction
) -> Callable[[Params, PhysicalConfiguration], jax.Array]:
    """Constructs ac_zvQ estimator [10.1063/1.1621615] of the HF force."""

    def evaluate_hf_force_ac_zvq_(
        params: Params, phys_conf: PhysicalConfiguration
    ) -> jax.Array:

        grad_Q = jax.jacfwd(Q)(phys_conf.r, phys_conf.R, hamil.mol.charges)
        grad_nuc_log_psi = make_grad_log_wf(wf)(params, phys_conf)
        force_nuc = coulomb_force(
            phys_conf.R, phys_conf.R, hamil.mol.charges, hamil.mol.charges, True
        )
        f_zv = (grad_nuc_log_psi[None, None] * grad_Q).sum((-1, -2)) + force_nuc
        return f_zv

    return evaluate_hf_force_ac_zvq_


def evaluate_hf_force_ac_zvzbq(
    hamil: MolecularHamiltonian, wf: ParametrizedWaveFunction
) -> Callable[[Params, PhysicalConfiguration, Energy, Energy], jax.Array]:
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
