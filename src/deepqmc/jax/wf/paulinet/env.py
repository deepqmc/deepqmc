import haiku as hk
import jax.numpy as jnp


class ExponentialEnvelopes(hk.Module):
    def __init__(self, centers, shells):
        super().__init__()
        self.centers = centers
        center_idx, zetas = zip(*shells)
        self.center_idx = jnp.array(center_idx)
        self.zetas = jnp.array(zetas)

    def __call__(self, diffs):
        return jnp.exp(-self.zetas * jnp.sqrt(diffs[..., self.center_idx, -1]))
