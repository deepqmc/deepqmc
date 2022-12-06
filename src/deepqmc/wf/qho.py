import haiku as hk
import jax
import jax.numpy as jnp

from ..types import Psi

__all__ = ()


class QHOAnsatz(hk.Module):
    def __init__(self, hamil):
        super().__init__()
        self.width = 1 / jnp.sqrt(hamil.mass * hamil.nu / 2)
        self.kernel = hk.nets.MLP([64, 32, 1], activation=jax.nn.silu)

    def __call__(self, r, return_mos=False):
        r = r / self.width
        x = self.kernel(r)
        x = jnp.squeeze(x, axis=-1)
        env = -jnp.sqrt(1 + jnp.sum(r**2))
        psi = Psi(jnp.sign(x), env + jnp.log(jnp.abs(x)))
        return psi
