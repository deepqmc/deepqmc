import jax
import jax.numpy as jnp

__all__ = ()


def laplacian(f):
    def lap(x):
        grad_f = jax.grad(f)
        df, grad_f_jvp = jax.linearize(grad_f, x)
        eye = jnp.eye(len(x))
        d2f = jnp.diag(jax.vmap(grad_f_jvp)(eye))
        return jnp.sum(d2f), df

    return lap


def masked_mean(x, mask):
    x = jnp.where(mask, x, 0)
    return x.sum() / jnp.sum(mask)
