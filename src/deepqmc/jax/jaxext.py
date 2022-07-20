import jax.numpy as jnp
from jax.nn import softplus


def SSP(x):
    return softplus(x) + jnp.log(0.5)
