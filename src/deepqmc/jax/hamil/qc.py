import jax.numpy as jnp
from jax import random

from ..physics import (
    electronic_potential,
    laplacian,
    nuclear_energy,
    nuclear_potential,
)

all__ = ()


class MolecularHamiltonian:
    def __init__(self, mol):
        self.mol = mol

    def init_sample(self, rng, n, elec_std=1.0):
        r"""
        Guess some initial electron positions.

        Tries to make an educated guess about plausible initial electron
        configurations. Places electrons according to normal distributions
        centered on the nuclei. If the molecule is not neutral, extra electrons
        are placed on or removed from random nuclei. The resulting configurations
        are usually very crude, a subsequent, thorough equilibration is needed.

        Args:
            rng (jax.random.PRNGKey): key used for PRNG
            n (int): the number of configurations to generate
            elec_std (float): optional, a factor for scaling the spread of
                electrons around the nuclei
        """
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
        def loc_ene(state, r, mol=self.mol):
            lap_log_psis, quantum_force = laplacian(
                lambda r: wf(state, r.reshape((-1, 3))).log
            )(r.flatten())
            Es_kin = -0.5 * (lap_log_psis + (quantum_force**2).sum(axis=-1))
            Es_nuc = nuclear_energy(mol)
            Vs_nuc = nuclear_potential(r, mol)
            Vs_el = electronic_potential(r)
            Es_loc = Es_kin + Vs_nuc + Vs_el + Es_nuc
            result = (Es_loc, quantum_force) if return_grad else Es_loc
            stats = {'hamil/V_el': Vs_el, 'hamil/E_kin': Es_kin, 'hamil/V_nuc': Vs_nuc}
            return result, stats

        return loc_ene
