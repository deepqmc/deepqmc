import math
from copy import deepcopy
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from .ecp.gaussian_type_ecp import GaussianTypeECP
from .geom import pairwise_distance
from .geom.coordinate_transform import (
    CartesianCoordinateTransform,
    InvertibleCoordinateTransform,
)
from .hamil import MolecularHamiltonian
from .physics import nuclear_energy, reverse_forward_laplacian
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


def make_general_jvp_nuc_wf(
    wf: ParametrizedWaveFunction,
    coordinate_transform: InvertibleCoordinateTransform,
):
    """Construct the derivative of the WF wrt. general nuclear coordinates using JVP."""

    def general_jvp_fn(params, phys_conf: PhysicalConfiguration, tangent: jax.Array):
        def jvp_wrapper(transformed_coords):
            R = coordinate_transform.to_cartesian(transformed_coords)
            return wf(params, jdc.replace(phys_conf, R=R)).log

        transformed_coords = coordinate_transform.from_cartesian(phys_conf.R)
        log_psi, grad_log_psi = jax.jvp(jvp_wrapper, (transformed_coords,), (tangent,))
        # log(Psi') = log(Psi) + log((log(Psi))')
        return Psi(jnp.zeros_like(log_psi), log_psi + jnp.log(jnp.abs(grad_log_psi)))

    return general_jvp_fn


def make_general_grad_fn(
    fn: Callable, coordinate_transform: InvertibleCoordinateTransform
):
    """Construct the gradient of a function wrt. general nuclear coordinates."""

    @jax.grad
    def transformed_grad_fn(transformed_coords, *args, phys_conf):
        R = coordinate_transform.to_cartesian(transformed_coords)
        return fn(*args, jdc.replace(phys_conf, R=R))

    def general_grad_fn(*args, phys_conf: PhysicalConfiguration) -> jax.Array:
        transformed_coords = coordinate_transform.from_cartesian(phys_conf.R)
        return transformed_grad_fn(transformed_coords, *args, phys_conf=phys_conf)

    return general_grad_fn


def make_grad_nuc_wf(wf: ParametrizedWaveFunction, i=None, j=None):
    """Constructs the grad of the wf wrt. nuclei.

    WARNING: This doesn't work with general coordinates yet. It throws an
    error, doesn't fail silently.
    """

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


def Q(
    r: jax.Array,
    R: jax.Array,
    c: jax.Array,
    coordinate_transform: InvertibleCoordinateTransform,
) -> jax.Array:
    """Constructs the Q function of [10.1063/1.1621615]."""
    dists = r[None] - R[:, None]
    force = c[:, None, None] * dists / jnp.linalg.norm(dists, axis=-1, keepdims=True)
    cartesian_Q = force.sum(-2)
    return coordinate_transform.from_cartesian(cartesian_Q)


def make_zv_term_via_jvp(
    hamil: MolecularHamiltonian,
    wf: ParametrizedWaveFunction,
    coordinate_transform: InvertibleCoordinateTransform,
):
    """Constructs the ZV term of the AC force estimators using jax.jvp."""
    wf_nuc_jvp = make_general_jvp_nuc_wf(wf, coordinate_transform)
    loop_hamil = deepcopy(hamil)
    loop_hamil.lap_factory = (
        reverse_forward_laplacian  # make sure not to use folx due to bugs
    )

    def zv_term_via_jvp(
        params: Params,
        phys_conf: PhysicalConfiguration,
        e_loc: Energy,
        grad_log_psi: jax.Array,
    ) -> jax.Array:
        def local_energy_grad_wf_i(carry, R_tangent) -> tuple[None, jax.Array]:
            eloc_ij, _ = loop_hamil.local_energy(
                partial(wf_nuc_jvp, tangent=R_tangent)
            )(None, params, phys_conf)
            return carry, eloc_ij

        transformed_coords = coordinate_transform.from_cartesian(phys_conf.R)
        R_tangents = jnp.eye(transformed_coords.size).reshape(
            -1, *transformed_coords.shape
        )
        _, e_loc_grad_psi = jax.lax.scan(local_energy_grad_wf_i, None, R_tangents)
        f_zv = (
            -(e_loc_grad_psi.reshape(transformed_coords.shape) - e_loc) * grad_log_psi
        )
        return f_zv

    return zv_term_via_jvp


def make_bare_plus_zvq_term(
    hamil: MolecularHamiltonian,
    wf: ParametrizedWaveFunction,
    coordinate_transform: InvertibleCoordinateTransform,
):
    """Constructs the sum of the bare and ZVQ terms of AC force estimators."""
    nuclear_force_fn = make_general_grad_fn(
        lambda pc: -nuclear_energy(pc, hamil.mol.charges), coordinate_transform
    )

    def bare_plus_zvq_term(
        phys_conf: PhysicalConfiguration, grad_log_psi: jax.Array
    ) -> jax.Array:
        grad_Q = jax.jacfwd(Q)(
            phys_conf.r, phys_conf.R, hamil.mol.charges, coordinate_transform
        )
        force_nuc = nuclear_force_fn(phys_conf=phys_conf)
        f_bare_plus_zvq = (
            jnp.expand_dims(grad_log_psi, range(grad_Q.ndim - 2)) * grad_Q
        ).sum((-1, -2)) + force_nuc
        return f_bare_plus_zvq

    return bare_plus_zvq_term


# Currently not used but could help redeuce fluctuations
# of the bare pp estiamtor
def vnl_ongradpsi(hamil: MolecularHamiltonian, wf: ParametrizedWaveFunction):
    """Constructs the non local potential acting on the wf gradient.

    WARNING: This doesn't work with general coordinates yet.
    """
    n_nuc = len(hamil.mol.coords)

    def vnl_ongradpsi_(
        rng: KeyArray, phys_conf: PhysicalConfiguration, params: Params
    ) -> jax.Array:
        def vnl_ongradpsi_ij(i: int, val: jax.Array) -> jax.Array:
            wfgrad_ij = partial(make_grad_nuc_wf(wf, i // 3, i % 3), params)
            v_nl_ongrad_ij = hamil.pot.nonloc_potential(rng, phys_conf, wfgrad_ij)
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


def evaluate_hf_force_bare(
    hamil: MolecularHamiltonian,
    wf: ParametrizedWaveFunction,
    coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
):
    """Constructs bare estimator of the HF force."""
    if coordinate_transform is None:
        coordinate_transform = CartesianCoordinateTransform(hamil.n_nuc)
    charges_nuc = hamil.pot.ns_valence
    nuclear_force_fn = make_general_grad_fn(
        lambda pc: -nuclear_energy(pc, charges_nuc), coordinate_transform
    )
    electronic_force_fn = make_general_grad_fn(
        lambda pc: -hamil.pot.local_potential(pc), coordinate_transform
    )

    def evaluate_hf_force_bare_(
        rng: KeyArray, params: Params, phys_conf: PhysicalConfiguration
    ) -> jax.Array:
        force_nuc = nuclear_force_fn(phys_conf=phys_conf)
        force_elec = electronic_force_fn(phys_conf=phys_conf)

        if isinstance(hamil.pot, GaussianTypeECP):
            assert isinstance(
                coordinate_transform, CartesianCoordinateTransform
            ), 'ECP forces are only implemented with CartesianCoordinateTransform'
            non_loc_force = -hamil.pot.grad_nonloc_potential(wf, rng, phys_conf, params)
            force_elec += non_loc_force.flatten()

        return force_nuc + force_elec

    return evaluate_hf_force_bare_


def evaluate_hf_force_ac_zv(
    hamil: MolecularHamiltonian,
    wf: ParametrizedWaveFunction,
    coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
):
    """Constructs ac_zv estimator [10.1063/5.0052266] of the HF force."""
    if coordinate_transform is None:
        coordinate_transform = CartesianCoordinateTransform(hamil.n_nuc)
    zv_term = make_zv_term_via_jvp(hamil, wf, coordinate_transform)
    grad_nuc_log_wf = make_general_grad_fn(
        (lambda params, pc: wf(params, pc).log), coordinate_transform
    )

    def evaluate_hf_force_ac_zv_(
        rng: KeyArray,
        params: Params,
        phys_conf: PhysicalConfiguration,
        e_loc: Optional[Energy] = None,
        energy: Optional[Energy] = None,
    ) -> jax.Array:
        f_bare = evaluate_hf_force_bare(hamil, wf, coordinate_transform)(
            rng, params, phys_conf
        )
        if e_loc is None:
            e_loc, _ = hamil.local_energy(wf)(rng, params, phys_conf)
        grad_nuc_log_psi = grad_nuc_log_wf(params, phys_conf=phys_conf)
        f_zv = zv_term(params, phys_conf, e_loc, grad_nuc_log_psi)

        return f_bare + f_zv

    return evaluate_hf_force_ac_zv_


def evaluate_hf_force_ac_zvzb(
    hamil: MolecularHamiltonian,
    wf: ParametrizedWaveFunction,
    coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
):
    """Constructs ac_zvzb estimator [10.1063/5.0052266] of the HF force."""

    if coordinate_transform is None:
        coordinate_transform = CartesianCoordinateTransform(hamil.n_nuc)
    zv_term = make_zv_term_via_jvp(hamil, wf, coordinate_transform)
    grad_nuc_log_wf = make_general_grad_fn(
        (lambda params, pc: wf(params, pc).log), coordinate_transform
    )

    def evaluate_hf_force_ac_zvzb_(
        rng: KeyArray,
        params: Params,
        phys_conf: PhysicalConfiguration,
        e_loc: Energy,
        energy: Energy,
    ) -> jax.Array:
        f_bare = evaluate_hf_force_bare(hamil, wf, coordinate_transform)(
            rng, params, phys_conf
        )
        grad_nuc_log_psi = grad_nuc_log_wf(params, phys_conf=phys_conf)
        f_zv = zv_term(params, phys_conf, e_loc, grad_nuc_log_psi)
        f_zb = (
            -2
            * jnp.expand_dims(e_loc - energy, range(grad_nuc_log_psi.ndim))
            * grad_nuc_log_psi
        )

        return f_bare + f_zv + f_zb

    return evaluate_hf_force_ac_zvzb_


def evaluate_hf_force_ac_zb(
    hamil: MolecularHamiltonian,
    wf: ParametrizedWaveFunction,
    coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
):
    """Constructs ac_zvzb estimator [10.1063/5.0052266] of the HF force."""
    if coordinate_transform is None:
        coordinate_transform = CartesianCoordinateTransform(hamil.n_nuc)
    grad_nuc_log_wf = make_general_grad_fn(
        (lambda params, pc: wf(params, pc).log), coordinate_transform
    )

    def evaluate_hf_force_ac_zb_(
        rng: KeyArray,
        params: Params,
        phys_conf: PhysicalConfiguration,
        e_loc: Energy,
        energy: Energy,
    ) -> jax.Array:
        f_bare = evaluate_hf_force_bare(hamil, wf)(rng, params, phys_conf)
        grad_nuc_log_psi = grad_nuc_log_wf(params, phys_conf=phys_conf)
        f_zb = (
            -2
            * jnp.expand_dims(e_loc - energy, range(grad_nuc_log_psi.ndim))
            * grad_nuc_log_psi
        )
        return f_bare + f_zb

    return evaluate_hf_force_ac_zb_


def evaluate_hf_force_ac_zvq(
    hamil: MolecularHamiltonian,
    wf: ParametrizedWaveFunction,
    coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
):
    """Constructs ac_zvQ estimator [10.1063/1.1621615] of the HF force."""

    if coordinate_transform is None:
        coordinate_transform = CartesianCoordinateTransform(hamil.n_nuc)
    bare_plus_zvq = make_bare_plus_zvq_term(hamil, wf, coordinate_transform)
    grad_log_wf = make_grad_log_wf(wf)

    def evaluate_hf_force_ac_zvq_(
        params: Params, phys_conf: PhysicalConfiguration
    ) -> jax.Array:
        grad_log_psi = grad_log_wf(params, phys_conf)
        f_bare_plus_zvq = bare_plus_zvq(phys_conf, grad_log_psi)
        return f_bare_plus_zvq

    return evaluate_hf_force_ac_zvq_


def evaluate_hf_force_ac_zvzbq(
    hamil: MolecularHamiltonian,
    wf: ParametrizedWaveFunction,
    coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
):
    """Constructs ac_zvzbQ estimator [10.1063/1.1621615] of the HF force."""

    if coordinate_transform is None:
        coordinate_transform = CartesianCoordinateTransform(hamil.n_nuc)
    bare_plus_zvq = make_bare_plus_zvq_term(hamil, wf, coordinate_transform)
    grad_log_wf = make_grad_log_wf(wf)

    def evaluate_hf_force_ac_zvzbq_(
        params: Params, phys_conf: PhysicalConfiguration, e_loc: Energy, energy: Energy
    ) -> jax.Array:
        grad_log_psi = grad_log_wf(params, phys_conf)
        f_bare_plus_zvq = bare_plus_zvq(phys_conf, grad_log_psi)
        f_zbq = (
            -2
            * (e_loc - energy)
            * Q(phys_conf.r, phys_conf.R, hamil.mol.charges, coordinate_transform)
        )
        return f_bare_plus_zvq + f_zbq

    return evaluate_hf_force_ac_zvzbq_


def evaluate_finite_difference_force(
    hamil: MolecularHamiltonian, wf: ParametrizedWaveFunction, step_size: float
):
    """Evaluates the inter atomic force based on a finite difference scheme."""

    def evaluate_finite_difference_force_(
        rng: KeyArray,
        params: Params,
        phys_conf: PhysicalConfiguration,
        e_loc: Energy,
        energy: Energy,
    ) -> jax.Array:

        shape = phys_conf.R.shape
        dR = (jnp.eye(math.prod(shape)) * step_size).reshape(-1, *shape)
        Rs = phys_conf.R[None] - dR
        dists = pairwise_distance(phys_conf.R, phys_conf.r)
        dr = jnp.einsum(
            'inj, ne -> iej',
            dR,
            (jnp.exp(-dists) / jnp.exp(-dists).sum(-2, keepdims=True)),
        )
        psi = wf(params, phys_conf)
        rs = phys_conf.r[None] + dr
        phys_confs = jdc.replace(
            phys_conf, R=Rs, r=rs, mol_idx=phys_conf.mol_idx.repeat(len(Rs))
        )
        psis = jax.lax.map(lambda pc: wf(params, pc), phys_confs)
        local_energies, _ = jax.lax.map(
            lambda pc: hamil.local_energy(wf)(rng, params, pc), phys_confs
        )
        weight = jnp.exp(2 * (psis.log - psi.log[None])).reshape(shape)
        finite_diff_force = ((local_energies - e_loc[None]) / step_size).reshape(shape)
        return weight * finite_diff_force

    return evaluate_finite_difference_force_


def evaluate_hf_force_ac_zvqzb(
    hamil: MolecularHamiltonian,
    wf: ParametrizedWaveFunction,
    coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
):
    """Constructs the hybrid ZVQ + full ZB estimator [10.1063/1.1621615]."""

    if coordinate_transform is None:
        coordinate_transform = CartesianCoordinateTransform(hamil.n_nuc)
    bare_plus_zvq = make_bare_plus_zvq_term(hamil, wf, coordinate_transform)
    grad_nuc_log_wf = make_general_grad_fn(
        (lambda params, pc: wf(params, pc).log), coordinate_transform
    )
    grad_log_wf = make_grad_log_wf(wf)

    def evaluate_hf_force_ac_zvqzb_(
        params: Params, phys_conf: PhysicalConfiguration, e_loc: Energy, energy: Energy
    ) -> jax.Array:
        grad_nuc_log_psi = grad_nuc_log_wf(params, phys_conf=phys_conf)
        grad_log_psi = grad_log_wf(params, phys_conf)
        f_bare_plus_zvq = bare_plus_zvq(phys_conf, grad_log_psi)
        f_zb = -2 * (e_loc - energy)[None, None] * grad_nuc_log_psi
        return f_bare_plus_zvq + f_zb

    return evaluate_hf_force_ac_zvqzb_
