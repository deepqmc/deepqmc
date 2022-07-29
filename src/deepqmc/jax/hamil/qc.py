import jax.numpy as jnp
from jax import random

from ..utils import (
    electronic_potential,
    laplacian,
    nuclear_energy,
    nuclear_potential,
)

all__ = ()


def laplacian_flat(f):
    return lambda r, **kwargs: laplacian(
        lambda r, **kwargs: f(r.reshape((-1, 3)), **kwargs).log
    )(r.flatten(), **kwargs)


class MolecularHamiltonian:
    def __init__(self, mol):
        self.mol = mol

    def init_sample(self, rng, n, elec_std=1.0):
        rng_remainder, rng_normal = random.split(rng)
        charges = self.mol.charges - self.mol.charge / len(self.mol.charges)
        base = jnp.floor(charges).astype(jnp.int32)
        prob = charges - base
        n_remainder = int(self.mol.charges.sum() - self.mol.charge - base.sum())
        idxs = jnp.tile(
            jnp.concatenate(
                [i_atom * jnp.ones(b, jnp.int32) for i_atom, b in enumerate(base)]
            )[None],
            (n, 1),
        )
        if n_remainder > 0:
            extra = random.categorical(rng_remainder, prob, shape=(n, n_remainder))
            idxs = jnp.concatenate([idxs, extra], axis=-1)

        centers = self.mol.coords[idxs]
        std = elec_std * jnp.sqrt(self.mol.charges)[idxs][..., None]
        rs = centers + std * random.normal(rng_normal, centers.shape)
        return rs

    def local_energy(self, wf, return_grad=False):
        def loc_ene(r, graph_edges, mol=self.mol):
            lap_log_psis, quantum_force = laplacian_flat(wf)(r, graph_edges=graph_edges)
            Es_kin = -0.5 * (lap_log_psis + (quantum_force**2).sum(axis=(-1)))
            Es_nuc = nuclear_energy(mol)
            Vs_nuc = nuclear_potential(r, mol)
            Vs_el = electronic_potential(r)
            Es_loc = Es_kin + Vs_nuc + Vs_el + Es_nuc
            result = (Es_loc, quantum_force) if return_grad else Es_loc
            return result

        return loc_ene
