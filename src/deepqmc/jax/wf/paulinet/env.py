import haiku as hk
import jax.numpy as jnp


class ExponentialEnvelopes(hk.Module):
    def __init__(self, shells):
        super().__init__()
        center_idx, zetas = zip(*shells)
        self.center_idx = jnp.array(center_idx)
        self.zetas = hk.get_parameter(
            'zetas', [len(zetas)], init=lambda shape, dtype: jnp.array(zetas)
        )

    def __call__(self, diffs):
        return jnp.exp(-jnp.abs(self.zetas) * jnp.sqrt(diffs[..., self.center_idx, -1]))

    @classmethod
    def from_mol(cls, mol):
        shells = []
        for i, z in enumerate(mol.charges):
            # find number of occupied shells for atom
            max_elec = 0
            n_shells = 0
            for n in range(10):
                if z <= max_elec:
                    break
                else:
                    n_shells += 1
                    for m in range(n + 1):
                        max_elec += 2 * (2 * m + 1)
            # adding the lowest unoccupied shell might be beneficial,
            # especially for transition metals
            #  n_shells += 1
            for k in range(n_shells):
                shells.append((i, z / (k + 1)))
        return cls(shells)
