import haiku as hk
import jax.numpy as jnp

from ...utils import norm


class ExponentialEnvelopes(hk.Module):
    r"""Create exponential envelopes centered on the nuclei."""

    def __init__(self, shells):
        super().__init__()
        center_idx, zetas = zip(*shells)
        self.center_idx = jnp.array(center_idx)
        self.zetas = hk.get_parameter(
            'zetas',
            jnp.array(zetas).shape,
            init=lambda shape, dtype: jnp.array(zetas),
        )

    def __call__(self, diffs):
        return jnp.exp(
            -norm(
                jnp.einsum(
                    'ers,ies->ier', self.zetas, diffs[..., self.center_idx, :-1]
                ),
                safe=True,
            )
        )

    @classmethod
    def from_mol(cls, mol):
        r"""Create the input of the constructor from a :class:`~deepqmc.Molecule`."""
        shells = []
        for i, (z, n_shell, n_pp_shell) in enumerate(
            zip(mol.charges, mol.n_shells, mol.n_pp_shells)
        ):
            for k in range(n_pp_shell, n_shell):
                #  shells.append((i, [[z / (k + 1)] * 3] * 3))
                shells.append((i, z / (k + 1) * jnp.eye(3)))
        return cls(shells)
