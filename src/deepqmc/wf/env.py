import haiku as hk
import jax.numpy as jnp
from jax.nn import softplus
from kfac_jax import register_scale_and_shift

from ..physics import pairwise_diffs
from ..utils import norm, unflatten


class ExponentialEnvelopes(hk.Module):
    r"""Create exponential envelopes centered on the nuclei."""

    def __init__(
        self,
        hamil,
        n_determinants,
        *,
        isotropic,
        per_shell,
        per_orbital_exponent,
        spin_restricted,
        init_to_ones,
        softplus_zeta,
    ):
        super().__init__()
        shells = []
        for i, (z, n_shell, n_ecp_shell) in enumerate(
            zip(hamil.mol.charges, hamil.mol_shells, hamil.mol_ecp_shells)
        ):
            for k in range(n_ecp_shell, n_shell if per_shell else n_ecp_shell + 1):
                shells.append((i, z / (k + 1)))
        self.center_idx, zetas = map(jnp.array, zip(*shells))  # [n_env]
        self.init_to_ones = init_to_ones
        self.pi = [
            self.get_pi_for_one_spin(
                name, n_determinants, hamil.n_up, hamil.n_down, len(zetas)
            )
            for name in (['pi'] if spin_restricted else ['pi_up', 'pi_down'])
        ]  # [n_orb, n_env]
        if per_orbital_exponent:
            zetas = jnp.tile(
                zetas[None], (n_determinants * (hamil.n_up + hamil.n_down), 1)
            )  # [n_orb, n_env]
        if not isotropic:
            zetas = zetas[..., None, None] * jnp.eye(3)
        self.zetas = [
            self.get_zeta_for_one_spin(name, zetas)
            for name in (['zetas'] if spin_restricted else ['zetas_up', 'zetas_down'])
        ]  # [n_env] or [n_orb, n_env] or [n_env, 3, 3] or [n_orb, n_env, 3, 3]
        self.isotropic = isotropic
        self.per_orbital_exponent = per_orbital_exponent
        self.spin_restricted = spin_restricted
        self.n_up = hamil.n_up
        self.n_det = n_determinants
        self.softplus_zeta = softplus_zeta

    def _call_for_one_spin(self, zeta, pi, diffs):
        d = diffs[..., self.center_idx, :-1]
        if self.isotropic:
            d = norm(d, safe=True)  # [n_el, n_env]
            if self.per_orbital_exponent:
                d = d[:, None]  # [n_el, 1, n_env]
            exponent = (
                (softplus(zeta) * d) if self.softplus_zeta else jnp.abs(zeta * d)
            )  # [n_el, n_env] or [n_el, n_orb, n_env]
            if self.softplus_zeta:
                exponent = register_scale_and_shift(exponent, d, scale=zeta, shift=None)
        else:
            exponent = norm(
                jnp.einsum('...ers,ies->i...er', zeta, d), safe=True
            )  # [n_el, n_env] or [n_el, n_orb, n_env]
        if not self.per_orbital_exponent:
            exponent = exponent[:, None]  # [n_el, 1, n_env]
        orbs = (pi * jnp.exp(-exponent)).sum(axis=-1)  # [n_el, n_orb]
        return unflatten(orbs, -1, (self.n_det, -1)).swapaxes(-2, -3)

    def get_pi_for_one_spin(self, name, n_determinants, n_up, n_down, n_env):
        return hk.get_parameter(
            name,
            (n_determinants * (n_up + n_down), n_env),
            init=lambda s, d: jnp.ones(s)
            + (0 if self.init_to_ones else hk.initializers.VarianceScaling(1.0)(s, d)),
        )

    def get_zeta_for_one_spin(self, name, zeta):
        return hk.get_parameter(
            name,
            zeta.shape,
            init=lambda shape, dtype: (
                jnp.ones(shape) if self.init_to_ones else jnp.copy(zeta)
            ),
        )

    def __call__(self, phys_conf, nuc_params):
        diffs = pairwise_diffs(phys_conf.r, phys_conf.R)
        if self.spin_restricted:
            return self._call_for_one_spin(self.zetas[0], self.pi[0], diffs)
        else:
            orbs = [
                self._call_for_one_spin(zeta, pi, diff)
                for zeta, pi, diff in zip(
                    # [nuc_params['zeta_up'], nuc_params['zeta_down']],
                    self.zetas,
                    self.pi,
                    jnp.split(diffs, (self.n_up,)),
                )
            ]
            return jnp.concatenate(orbs, axis=-2)


class SimplifiedNucleusDependentEnvelopes(hk.Module):
    r"""Envelopes with no trainable pi parameter."""

    def __init__(
        self,
        hamil,
        n_determinants,
        *,
        n_envelope_per_nucleus,
        per_orbital_exponent,
        fixed_pi,
    ):
        super().__init__()
        self.n_determinants = n_determinants
        self.n_up = hamil.n_up
        self.n_env_per_nuc = n_envelope_per_nucleus
        n_nuc = len(hamil.mol.charges)
        self.n_nuc = n_nuc
        n_orb = hamil.n_up + hamil.n_down
        self.n_orb = n_orb
        self.n_det = n_determinants
        self.per_orbital_exponent = per_orbital_exponent
        self.fixed_pi = fixed_pi
        self.pis = (
            jnp.ones((n_nuc, n_orb, n_determinants, n_envelope_per_nucleus)),
            jnp.ones((n_nuc, n_orb, n_determinants, n_envelope_per_nucleus)),
        )

    def call_for_one_spin(self, zeta, pi, distance):
        # [n_el, n_nuc, 1, n_det, n_env_per_nuc] or
        # [n_el, n_nuc, n_orb, n_det, n_env_per_nuc] if per_orbital
        exponent = (
            jnp.abs(distance * zeta)
            if self.per_orbital_exponent
            else jnp.abs(distance * zeta)[:, :, None]
        )
        return jnp.moveaxis(
            (pi * jnp.exp(-exponent)).sum(axis=(1, -1)), 2, 0
        )  # [n_det, n_el, n_orb]

    def reshape_parameters(self, zeta, orbital_dimension):
        return zeta.reshape(
            (-1, self.n_orb, self.n_det, self.n_env_per_nuc)
            if orbital_dimension
            else (-1, self.n_det, self.n_env_per_nuc)
        )

    def __call__(self, phys_conf, nuc_params):
        diffs = pairwise_diffs(phys_conf.r, phys_conf.R)
        distance = jnp.tile(
            norm(diffs[:, :, :-1], safe=True)[:, :, None, None],
            (1, 1, self.n_determinants, self.n_env_per_nuc),
        )  # [n_el, n_nuc, n_det, n_env_per_nuc]
        if self.per_orbital_exponent:
            distance = jnp.tile(
                distance[:, :, None, :, :], (1, 1, self.n_orb, 1, 1)
            )  # [n_el, n_nuc, n_orb, n_det, n_env_per_nuc]
        if nuc_params is None:
            zetas = (
                hk.get_parameter(
                    'zetas_up',
                    (
                        (self.n_nuc, self.n_orb, self.n_det, self.n_env_per_nuc)
                        if self.per_orbital_exponent
                        else (self.n_nuc, self.n_det, self.n_env_per_nuc)
                    ),
                ),
                hk.get_parameter(
                    'zetas_down',
                    (
                        (self.n_nuc, self.n_orb, self.n_det, self.n_env_per_nuc)
                        if self.per_orbital_exponent
                        else (self.n_nuc, self.n_det, self.n_env_per_nuc)
                    ),
                ),
            )
        else:
            zetas = (
                self.reshape_parameters(
                    nuc_params['zetas_up'], self.per_orbital_exponent
                ),
                self.reshape_parameters(
                    nuc_params['zetas_down'], self.per_orbital_exponent
                ),
            )
        pis = (
            self.pis
            if self.fixed_pi
            else (
                self.reshape_parameters(nuc_params['pis_up'], True),
                self.reshape_parameters(nuc_params['pis_down'], True),
            )
        )
        return jnp.concatenate(
            [
                self.call_for_one_spin(zeta, pi, dist)
                for zeta, pi, dist in zip(zetas, pis, jnp.split(distance, (self.n_up,)))
            ],
            axis=-2,
        )
