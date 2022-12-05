import haiku as hk
import jax.numpy as jnp


class ExponentialEnvelopes(hk.Module):
    r"""Create exponential envelopes centered on the nuclei."""

    def __init__(self, shells):
        super().__init__()
        center_idx, zetas = zip(*shells)
        self.center_idx = jnp.array(center_idx)
        self.zetas = hk.get_parameter(
            'zetas', [len(zetas)], init=lambda shape, dtype: jnp.array(zetas)
        )

    def __call__(self, diffs):
        return jnp.exp(-jnp.abs(self.zetas * jnp.sqrt(diffs[..., self.center_idx, -1])))

    @classmethod
    def from_mol(cls, mol):
        r"""Create the input of the constructor from a :class:`~deepqmc.Molecule`."""
        shells = []
        for i, (z, n_shell) in enumerate(zip(mol.charges, mol.n_shells)):
            for k in range(n_shell):
                shells.append((i, z / (k + 1)))
        return cls(shells)
