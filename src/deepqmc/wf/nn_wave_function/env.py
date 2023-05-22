import haiku as hk
import jax.numpy as jnp

from ...utils import norm


class ExponentialEnvelopes(hk.Module):
    r"""Create exponential envelopes centered on the nuclei."""

    def __init__(
        self, mol, n_determinants, *, isotropic, per_shell, per_orbital_exponent
    ):
        super().__init__()
        shells = []
        for i, (z, n_shell, n_pp_shell) in enumerate(
            zip(mol.charges, mol.n_shells, mol.n_pp_shells)
        ):
            for k in range(n_pp_shell, n_shell if per_shell else n_pp_shell + 1):
                shells.append((i, z / (k + 1)))
        self.center_idx, zetas = map(jnp.array, zip(*shells))  # [n_env]
        self.pi = hk.get_parameter(
            'pi',
            (n_determinants * (mol.n_up + mol.n_down), len(zetas)),
            init=lambda s, d: hk.initializers.VarianceScaling(1.0)(s, d) + jnp.ones(s),
        )  # [n_orb, n_env]
        if per_orbital_exponent:
            zetas = jnp.tile(
                zetas[None], (n_determinants * (mol.n_up + mol.n_down), 1)
            )  # [n_orb, n_env]
        if not isotropic:
            zetas = zetas[..., None, None] * jnp.eye(3)
        self.zetas = hk.get_parameter(
            'zetas',
            zetas.shape,
            init=lambda shape, dtype: zetas,
        )  # [n_env] or [n_orb, n_env] or [n_env, 3, 3] or [n_orb, n_env, 3, 3]
        self.isotropic = isotropic
        self.per_orbital_exponent = per_orbital_exponent

    def __call__(self, diffs):
        d = diffs[..., self.center_idx, :-1]
        if self.isotropic:
            d = norm(d, safe=True)  # [n_el, n_env]
            if self.per_orbital_exponent:
                d = d[:, None]  # [n_el, 1, n_env]
            exponent = jnp.abs(self.zetas * d)  # [n_el, n_env] or [n_el, n_orb, n_env]
        else:
            exponent = norm(
                jnp.einsum('...ers,ies->i...er', self.zetas, d), safe=True
            )  # [n_el, n_env] or [n_el, n_orb, n_env]
        if not self.per_orbital_exponent:
            exponent = exponent[:, None]  # [n_el, 1, n_env]
        return (self.pi * jnp.exp(-exponent)).sum(axis=-1)  # [n_el, n_orb]
