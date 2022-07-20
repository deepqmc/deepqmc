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


def batch_laplacian(f, return_grad=False):
    def lap(rs):
        df = jax.grad(lambda x: jnp.sum(f(x)))
        d2f = jax.grad(lambda x: jnp.sum(df(x)))
        result = (jnp.sum(d2f(rs), axis=(-1, -2)),)
        if return_grad:
            result += (df(rs),)
        return result

    return lap


def masked_mean(x, mask):
    x = jnp.where(mask, x, 0)
    return x.sum() / jnp.sum(mask)


def pairwise_distance(coords1, coords2):
    return jnp.linalg.norm(coords1[..., :, None, :] - coords2[..., None, :, :], axis=-1)


def pairwise_diffs(coords1, coords2):
    diffs = coords1[..., :, None, :] - coords2[..., None, :, :]
    return jnp.concatenate([diffs, (diffs**2).sum(axis=-1, keepdims=True)], axis=-1)


def nuclear_energy(mol):
    coords, charges = mol.coords, mol.charges
    coulombs = (
        charges[:, None] * charges / jnp.linalg.norm(coords[:, None] - coords, axis=-1)
    )
    return jnp.triu(coulombs, 1).sum()


def nuclear_potential(rs, mol):
    dists = jnp.linalg.norm(rs[:, :, None] - mol.coords, axis=-1)
    return -(mol.charges / dists).sum(axis=(-1, -2))


def electronic_potential(rs):
    i, j = jnp.triu_indices(rs.shape[-2], k=1)
    dists = jnp.linalg.norm((rs[:, :, None] - rs[:, None, :])[:, i, j], axis=-1)
    return (1 / dists).sum(axis=-1)
