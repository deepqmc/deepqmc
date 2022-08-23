from functools import partial

import jax
import jax.numpy as jnp
from frozendict import frozendict

__all__ = ()


def freeze_dict(dictionary):
    new_dict = {}
    for k, v in dictionary.items():
        if isinstance(v, dict):
            v = freeze_dict(v)
        new_dict[k] = v
    return frozendict(new_dict)


def unfreeze_dict(dictionary):
    new_dict = {}
    for k, v in dictionary.items():
        if isinstance(v, frozendict):
            v = unfreeze_dict(v)
        new_dict[k] = v
    return dict(new_dict)


def laplacian_oneshot(f):
    def lap(x, **kwargs):
        _f = partial(f, **kwargs)
        grad_f = jax.grad(_f)
        df, grad_f_jvp = jax.linearize(grad_f, x)
        eye = jnp.eye(len(x))
        d2f = jnp.diag(jax.vmap(grad_f_jvp)(eye))
        return jnp.sum(d2f), df

    return lap


def laplacian(f):
    def lap(x, **kwargs):
        n_coord = len(x)
        _f = partial(f, **kwargs)
        grad_f = jax.grad(_f)
        df, grad_f_jvp = jax.linearize(grad_f, x)
        eye = jnp.eye(n_coord)
        d2f = lambda i, val: val + grad_f_jvp(eye[i])[i]
        d2f_sum = jax.lax.fori_loop(0, n_coord, d2f, 0.0)
        return d2f_sum, df

    return lap


def masked_mean(x, mask):
    x = jnp.where(mask, x, 0)
    return x.sum() / jnp.sum(mask)


def pairwise_distance(coords1, coords2):
    return jnp.linalg.norm(coords1[..., :, None, :] - coords2[..., None, :, :], axis=-1)


def pairwise_diffs(coords1, coords2):
    diffs = coords1[..., :, None, :] - coords2[..., None, :, :]
    return jnp.concatenate([diffs, (diffs**2).sum(axis=-1, keepdims=True)], axis=-1)


def pairwise_self_distance(coords, full=False):
    i, j = jnp.triu_indices(coords.shape[-2], k=1)
    diffs = coords[..., :, None, :] - coords[..., None, :, :]
    dists = jnp.linalg.norm(diffs[..., i, j, :], axis=-1)
    if full:
        dists_full = jnp.zeros(diffs.shape[:-1])
        dists_full = dists_full.at[..., i, j].set(dists)
        dists_full = dists_full.at[..., j, i].set(dists)
        dists = dists_full
    return dists


def triu_flat(x):
    i, j = jnp.triu_indices(x.shape[-1], 1)
    return x[..., i, j]


def nuclear_energy(mol):
    coords, charges = mol.coords, mol.charges
    coulombs = (
        charges[:, None] * charges / jnp.linalg.norm(coords[:, None] - coords, axis=-1)
    )
    return jnp.triu(coulombs, 1).sum()


def nuclear_potential(rs, mol):
    dists = jnp.linalg.norm(rs[..., :, None, :] - mol.coords, axis=-1)
    return -(mol.charges / dists).sum(axis=(-1, -2))


def electronic_potential(rs):
    i, j = jnp.triu_indices(rs.shape[-2], k=1)
    dists = jnp.linalg.norm(
        (rs[..., :, None, :] - rs[..., None, :, :])[..., i, j, :], axis=-1
    )
    return (1 / dists).sum(axis=-1)


@jax.jit
@jax.vmap
def vec_where(cond, x, y):
    return jnp.where(cond, x, y)
