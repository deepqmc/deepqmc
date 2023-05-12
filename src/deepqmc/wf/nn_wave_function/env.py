import haiku as hk
import jax.numpy as jnp

from ...utils import norm


class ExponentialEnvelopes(hk.Module):
    r"""Create exponential envelopes centered on the nuclei."""

    def __init__(self, shells, *, isotropic):
        super().__init__()
        center_idx, zetas = map(jnp.array, zip(*shells))
        self.center_idx = center_idx
        if not isotropic:
            zetas = zetas[..., None, None] * jnp.eye(3)

        self.zetas = hk.get_parameter(
            'zetas',
            zetas.shape,
            init=lambda shape, dtype: zetas,
        )
        self.isotropic = isotropic

    def __call__(self, diffs):
        d = diffs[..., self.center_idx, :-1]
        exponent = (
            jnp.abs(self.zetas * norm(d, safe=True))
            if self.isotropic
            else norm(jnp.einsum('ers,ies->ier', self.zetas, d), safe=True)
        )
        return jnp.exp(-exponent)

    @classmethod
    def from_mol(cls, mol, **kwargs):
        r"""Create the input of the constructor from a :class:`~deepqmc.Molecule`."""
        shells = []
        for i, (z, n_shell, n_pp_shell) in enumerate(
            zip(mol.charges, mol.n_shells, mol.n_pp_shells)
        ):
            for k in range(n_pp_shell, n_shell):
                shells.append((i, z / (k + 1)))
        return cls(shells, **kwargs)
