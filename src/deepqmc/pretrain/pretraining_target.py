import logging
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
from jax.nn import one_hot

from ..physics import pairwise_diffs
from .gto import GTOBasis

log = logging.getLogger(__name__)

# TODO merge with ../wf/baseline/baseline.py


class PretrainTarget:
    r"""Represent an (MC-)SCF wave function, used as baseline."""

    def __init__(self, hamil, n_determinants, centers, shells, mo_coeffs):
        self.n_determinants = n_determinants
        basis = hk.without_apply_rng(
            hk.transform(lambda diffs: GTOBasis(centers, shells)(diffs))
        )
        basis_params = basis.init(
            jax.random.PRNGKey(0),
            jnp.zeros((hamil.n_up + hamil.n_down, centers.shape[0], 3)),
        )
        self.basis = partial(basis.apply, basis_params)
        self.mo_coeffs = mo_coeffs

    def __call__(self, confs, conf_coeffs, phys_conf):
        mol_idx = phys_conf.mol_idx
        diffs = pairwise_diffs(phys_conf.r, phys_conf.R)
        n_el = diffs.shape[-3]
        aos = self.basis(diffs)
        mos = jnp.matmul(aos, self.mo_coeffs[mol_idx])
        # Shape (n_det, n_elec, n_orb)
        mos = mos[:, confs[mol_idx]].swapaxes(0, 1)[: self.n_determinants]
        # ci coefficients are included in the orbitals of the respective determinant
        factors = (jnp.abs(conf_coeffs[mol_idx]) ** (1 / n_el))[:, None] * (
            one_hot(0, n_el)[None, :] * jnp.sign(conf_coeffs[mol_idx])[:, None]
            + (1 - one_hot(0, n_el)[None, :])
        )
        return mos * factors[: self.n_determinants, None, :]
