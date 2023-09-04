import jax
import jax.numpy as jnp

from .types import Potential
from .utils import norm, triu_flat

__all__ = ()


def pairwise_distance(coords1, coords2):
    return jnp.linalg.norm(coords1[..., :, None, :] - coords2[..., None, :, :], axis=-1)


def pairwise_diffs(coords1, coords2):
    diffs = coords1[..., :, None, :] - coords2[..., None, :, :]
    return jnp.concatenate([diffs, (diffs**2).sum(axis=-1, keepdims=True)], axis=-1)


def pairwise_self_distance(coords, full=False):
    i, j = jnp.triu_indices(coords.shape[-2], k=1)
    diffs = coords[..., :, None, :] - coords[..., None, :, :]
    dists = norm(diffs[..., i, j, :], safe=True, axis=-1)
    if full:
        dists = (
            jnp.zeros(diffs.shape[:-1])
            .at[..., i, j]
            .set(dists)
            .at[..., j, i]
            .set(dists)
        )
    return dists


def nuclear_energy(phys_conf, ns_valence):
    coulombs = triu_flat(ns_valence[:, None] * ns_valence) / pairwise_self_distance(
        phys_conf.R
    )
    return coulombs.sum()


def electronic_potential(phys_conf):
    dists = pairwise_self_distance(phys_conf.r)
    return (1 / dists).sum(axis=-1)


class NuclearCoulombPotential(Potential):
    """Class for the classical Coulomb potential."""

    def __init__(self, charges):
        self.charges = charges
        self.ns_valence = charges

    def local_potential(self, phys_conf):
        dists = pairwise_distance(phys_conf.r, phys_conf.R)
        return -(self.charges / dists).sum(axis=(-1, -2))


def laplacian(f):
    def lap(x):
        n_coord = len(x)
        grad_f = jax.grad(f)
        df, grad_f_jvp = jax.linearize(grad_f, x)
        eye = jnp.eye(n_coord)
        d2f = lambda i, val: val + grad_f_jvp(eye[i])[i]
        d2f_sum = jax.lax.fori_loop(0, n_coord, d2f, 0.0)
        return d2f_sum, df

    return lap
