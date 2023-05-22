import haiku as hk
import jax.numpy as jnp

from ...utils import norm

__all__ = ['ExponentialEnvelopes', 'FermiNetEnvelopes']


def get_shells(mol, isotropic, per_shell):
    shells = []
    for i, (z, n_shell, n_pp_shell) in enumerate(
        zip(mol.charges, mol.n_shells, mol.n_pp_shells)
    ):
        for k in range(n_pp_shell, n_shell if per_shell else n_pp_shell + 1):
            shells.append((i, z / (k + 1)))
    center_idx, zetas = map(jnp.array, zip(*shells))
    if not isotropic:
        zetas = zetas[..., None, None] * jnp.eye(3)
    return center_idx, zetas


class ExponentialEnvelopes(hk.Module):
    r"""Create exponential envelopes centered on the nuclei."""

    def __init__(self, mol, n_determinants, *, isotropic, per_shell):
        super().__init__()
        self.center_idx, zetas = get_shells(mol, isotropic, per_shell)
        self.zetas = hk.get_parameter(
            'zetas',
            zetas.shape,
            init=lambda shape, dtype: zetas,
        )
        self.isotropic = isotropic
        self.mo_coeff = hk.Linear(
            n_determinants * (mol.n_up + mol.n_down),
            with_bias=False,
            w_init=lambda s, d: hk.initializers.VarianceScaling(1.0)(s, d)
            + jnp.ones(s),
            name='mo_coeff',
        )

    def __call__(self, diffs):
        d = diffs[..., self.center_idx, :-1]
        exponent = (
            jnp.abs(self.zetas * norm(d, safe=True))
            if self.isotropic
            else norm(jnp.einsum('ers,ies->ier', self.zetas, d), safe=True)
        )
        return self.mo_coeff(jnp.exp(-exponent))


class FermiNetEnvelopes(hk.Module):
    r"""Create FermiNet type envelopes centered on the nuclei."""

    def __init__(self, mol, n_determinants, *, isotropic, per_shell):
        super().__init__()
        self.center_idx, zetas = get_shells(mol, isotropic, per_shell)
        zetas = jnp.tile(zetas[:, None], (1, n_determinants * (mol.n_up + mol.n_down)))
        self.zetas = hk.get_parameter(
            'zetas',
            zetas.shape,
            init=lambda shape, dtype: zetas,
        )
        self.isotropic = isotropic
        self.pi = hk.get_parameter(
            'pi',
            zetas.shape[:2],
            init=lambda s, d: hk.initializers.VarianceScaling(1.0)(s, d) + jnp.ones(s),
        )

    def __call__(self, diffs):
        d = diffs[..., self.center_idx, :-1]
        exponent = (
            jnp.abs(self.zetas * norm(d, safe=True)[..., None])
            if self.isotropic
            else norm(jnp.einsum('eors,ies->ieor', self.zetas, d), safe=True)
        )
        return (self.pi * jnp.exp(-exponent)).sum(axis=-2)
