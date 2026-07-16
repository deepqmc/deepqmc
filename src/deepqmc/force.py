import math
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import Optional

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
    """Wrap a force estimator with antithetic-sampling variance reduction.

    For each sample, mirrors the electrons that lie within ``r_cut`` of their nearest
    nucleus through that nucleus, evaluates ``evaluate_force`` on both the original
    and the mirrored configuration, and returns their importance-weighted average.
    This reduces the variance of the force estimator without introducing additional
    bias. Only compatible with estimators that do not require the local energy or the
    mean energy as an input, e.g. :func:`evaluate_hf_force_bare` or
    :func:`evaluate_hf_force_ac_zvq`.

    Args:
        evaluate_force (~collections.abc.Callable): a force estimator of signature
            ``(rng, params, phys_conf) -> jax.Array``, e.g. as returned by
            :func:`evaluate_hf_force_bare`.
        wf (~deepqmc.types.ParametrizedWaveFunction): the parametrized wave function.
        r_cut (float): the cutoff radius around each nucleus within which electrons
            are mirrored.

    Returns:
        ~collections.abc.Callable: a function of signature ``(rng, params, phys_conf)
        -> jax.Array`` that evaluates the antithetic-sampling force estimate for a
        batch of samples.
    """

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
    """Construct the bare estimator of the Hellmann-Feynman force.

    The bare estimator is the direct gradient of the local energy with respect to the
    nuclear coordinates, without any variance-reduction terms. It therefore has the
    largest variance among the estimators implemented in this module, but is also the
    cheapest to evaluate. If the Hamiltonian uses a Gaussian-type effective core
    potential, the non-local ECP contribution to the force is added as well.

    Args:
        hamil (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the system.
        wf (~deepqmc.types.ParametrizedWaveFunction): the parametrized wave function.
        coordinate_transform (~deepqmc.geom.coordinate_transform.InvertibleCoordinateTransform):
            optional, the coordinate system in which the force is expressed. Defaults
            to Cartesian nuclear coordinates.

    Returns:
        ~collections.abc.Callable: a function of signature ``(rng, params, phys_conf)
        -> jax.Array`` that evaluates the bare force for a batch of samples.
    """
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
    """Construct the AC-ZV (zero-variance) Hellmann-Feynman force estimator.

    Adds a zero-variance (ZV) correction term to :func:`evaluate_hf_force_bare`,
    computed via a JVP of the local energy through the nuclear coordinates
    [Tiihonen21]_. This reduces the variance of the estimator compared to the bare
    estimator, at the cost of an additional local-energy-gradient evaluation.

    Args:
        hamil (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the system.
        wf (~deepqmc.types.ParametrizedWaveFunction): the parametrized wave function.
        coordinate_transform (~deepqmc.geom.coordinate_transform.InvertibleCoordinateTransform):
            optional, the coordinate system in which the force is expressed. Defaults
            to Cartesian nuclear coordinates.

    Returns:
        ~collections.abc.Callable: a function of signature ``(rng, params, phys_conf,
        e_loc=None, energy=None) -> jax.Array`` that evaluates the AC-ZV force for a
        batch of samples. If ``e_loc`` is not provided it is computed internally;
        ``energy`` is accepted for interface uniformity with the other estimators but
        is not used.
    """
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
    """Construct the AC-ZVZB (zero-variance zero-bias) Hellmann-Feynman force estimator.

    Adds both the zero-variance (ZV) correction of :func:`evaluate_hf_force_ac_zv` and
    a zero-bias (ZB) correction to :func:`evaluate_hf_force_bare`. The ZB term
    corrects for the bias introduced by using finite Monte Carlo samples of a wave
    function that does not exactly satisfy the Schrödinger equation
    [Tiihonen21]_.

    Args:
        hamil (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the system.
        wf (~deepqmc.types.ParametrizedWaveFunction): the parametrized wave function.
        coordinate_transform (~deepqmc.geom.coordinate_transform.InvertibleCoordinateTransform):
            optional, the coordinate system in which the force is expressed. Defaults
            to Cartesian nuclear coordinates.

    Returns:
        ~collections.abc.Callable: a function of signature ``(rng, params, phys_conf,
        e_loc, energy) -> jax.Array`` that evaluates the AC-ZVZB force for a batch of
        samples, given the local energies ``e_loc`` and the mean energy ``energy`` of
        the batch.
    """

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
    """Construct the AC-ZB (zero-bias) Hellmann-Feynman force estimator.

    Adds only the zero-bias (ZB) correction term to :func:`evaluate_hf_force_bare`,
    without the zero-variance (ZV) term of :func:`evaluate_hf_force_ac_zv`
    [Tiihonen21]_. Cheaper to evaluate than :func:`evaluate_hf_force_ac_zvzb`, but
    with a higher variance since it lacks the ZV correction.

    Args:
        hamil (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the system.
        wf (~deepqmc.types.ParametrizedWaveFunction): the parametrized wave function.
        coordinate_transform (~deepqmc.geom.coordinate_transform.InvertibleCoordinateTransform):
            optional, the coordinate system in which the force is expressed. Defaults
            to Cartesian nuclear coordinates.

    Returns:
        ~collections.abc.Callable: a function of signature ``(rng, params, phys_conf,
        e_loc, energy) -> jax.Array`` that evaluates the AC-ZB force for a batch of
        samples, given the local energies ``e_loc`` and the mean energy ``energy`` of
        the batch.
    """
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
    """Construct the AC-ZVQ (zero-variance, closed-form) Hellmann-Feynman force estimator.

    Combines the bare nuclear and electronic force with a zero-variance correction
    expressed in closed form through the auxiliary function ``Q``, following
    [Assaraf03]_. Unlike :func:`evaluate_hf_force_ac_zv`, this estimator does not
    require an rng key or an extra local-energy evaluation, but it is not compatible
    with effective core potentials.

    Args:
        hamil (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the system.
        wf (~deepqmc.types.ParametrizedWaveFunction): the parametrized wave function.
        coordinate_transform (~deepqmc.geom.coordinate_transform.InvertibleCoordinateTransform):
            optional, the coordinate system in which the force is expressed. Defaults
            to Cartesian nuclear coordinates.

    Returns:
        ~collections.abc.Callable: a function of signature ``(params, phys_conf) ->
        jax.Array`` that evaluates the AC-ZVQ force for a batch of samples.
    """

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
    """Construct the AC-ZVZBQ (zero-variance zero-bias, closed-form) force estimator.

    Adds a zero-bias (ZB) correction, expressed via the auxiliary function ``Q``,
    to :func:`evaluate_hf_force_ac_zvq` [Assaraf03]_. Like the ZVQ estimator, this
    does not require an rng key, but it is not compatible with effective core
    potentials.

    Args:
        hamil (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the system.
        wf (~deepqmc.types.ParametrizedWaveFunction): the parametrized wave function.
        coordinate_transform (~deepqmc.geom.coordinate_transform.InvertibleCoordinateTransform):
            optional, the coordinate system in which the force is expressed. Defaults
            to Cartesian nuclear coordinates.

    Returns:
        ~collections.abc.Callable: a function of signature ``(params, phys_conf,
        e_loc, energy) -> jax.Array`` that evaluates the AC-ZVZBQ force for a batch of
        samples, given the local energies ``e_loc`` and the mean energy ``energy`` of
        the batch.
    """

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
    """Construct a finite-difference estimator of the interatomic force.

    Displaces each nuclear coordinate by :math:`\\pm` ``step_size`` (electron
    positions are co-displaced to follow the nearest nucleus) and estimates the force
    from the resulting change in the local energy, importance-weighted by the ratio
    of wave function values. Unlike the Hellmann-Feynman estimators in this module,
    this estimator does not require differentiating through the Hamiltonian, but
    its cost scales with the number of nuclear degrees of freedom, and its accuracy
    is limited by the finite step size.

    Args:
        hamil (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the system.
        wf (~deepqmc.types.ParametrizedWaveFunction): the parametrized wave function.
        step_size (float): the finite-difference step size, in Cartesian nuclear
            coordinates.

    Returns:
        ~collections.abc.Callable: a function of signature ``(rng, params, phys_conf,
        e_loc, energy) -> jax.Array`` that evaluates the finite-difference force for a
        batch of samples, given the local energies ``e_loc``. ``energy`` is accepted
        for interface uniformity with the other estimators but is not used.
    """

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
    """Construct the hybrid AC-ZVQ + ZB Hellmann-Feynman force estimator.

    Combines the closed-form zero-variance correction of
    :func:`evaluate_hf_force_ac_zvq` with a zero-bias correction computed via a
    general autodiff gradient of the log wave function with respect to the nuclear
    coordinates, instead of the closed-form ``Q`` function used in
    :func:`evaluate_hf_force_ac_zvzbq` [Assaraf03]_. Not compatible with effective
    core potentials.

    Args:
        hamil (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the system.
        wf (~deepqmc.types.ParametrizedWaveFunction): the parametrized wave function.
        coordinate_transform (~deepqmc.geom.coordinate_transform.InvertibleCoordinateTransform):
            optional, the coordinate system in which the force is expressed. Defaults
            to Cartesian nuclear coordinates.

    Returns:
        ~collections.abc.Callable: a function of signature ``(params, phys_conf,
        e_loc, energy) -> jax.Array`` that evaluates the hybrid AC-ZVQ + ZB force for
        a batch of samples, given the local energies ``e_loc`` and the mean energy
        ``energy`` of the batch.
    """

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
