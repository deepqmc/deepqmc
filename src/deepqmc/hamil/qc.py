from functools import partial

import jax.numpy as jnp
from jax import random, vmap

from ..physics import (
    electronic_potential,
    laplacian,
    nuclear_energy,
    nuclear_potential,
    pairwise_distance,
)
from ..utils import argmax_random_choice
from .base import Hamiltonian

__all__ = ['MolecularHamiltonian']


class MolecularHamiltonian(Hamiltonian):
    r"""
    Hamiltonian of non-relativistic molecular systems.

    The system consists of nuclei with fixed positions and electrons moving
    around them. The total energy is defined as the sum of the nuclear-nuclear
    and electron-electron repulsion, the nuclear-electron attraction, and the
    kinetic energy of the electrons:
    :math:`E=V_\text{nuc-nuc} + V_\text{el-el} + V_\text{nuc-el} + E_\text{kin}`.

    Args:
        mol (~deepqmc.Molecule): the molecule to consider
    """

    def __init__(self, *, mol):
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
            rng (jax.random.PRNGKey): key used for PRNG.
            n (int): the number of configurations to generate.
            elec_std (float): optional, a factor for scaling the spread of
                electrons around the nuclei.
        """

        rng_remainder, rng_normal, rng_spin = random.split(rng, 3)
        charges = self.mol.charges - self.mol.charge / len(self.mol.charges)
        base = jnp.floor(charges).astype(jnp.int32)
        prob = charges - base
        electrons_of_atom = jnp.tile(base[None], (n, 1))
        n_remainder = int(self.mol.charges.sum() - self.mol.charge - base.sum())
        if n_remainder > 0:
            extra = random.categorical(rng_remainder, prob, shape=(n, n_remainder))
            n_extra = vmap(partial(jnp.bincount, length=base.shape[-1]))(extra)
            electrons_of_atom += n_extra
        idxs = []
        rng_spin = random.split(rng_spin, len(electrons_of_atom))
        for rng_spin, elec_of_atom in zip(rng_spin, electrons_of_atom):
            up, down = self.distribute_spins(rng_spin, elec_of_atom)
            idxs.append(
                jnp.concatenate(
                    [
                        i_atom * jnp.ones(n_up, jnp.int32)
                        for i_atom, n_up in enumerate(up)
                    ]
                    + [
                        i_atom * jnp.ones(n_down, jnp.int32)
                        for i_atom, n_down in enumerate(down)
                    ]
                )
            )
        idxs = jnp.stack(idxs)
        centers = self.mol.coords[idxs]
        std = elec_std * jnp.sqrt(self.mol.charges)[idxs][..., None]
        rs = centers + std * random.normal(rng_normal, centers.shape)
        return rs

    def distribute_spins(self, rng, elec_of_atom):
        up, down = jnp.zeros_like(elec_of_atom), jnp.zeros_like(elec_of_atom)
        # try to distribute electron pairs evenly across atoms
        for i in range(jnp.max(elec_of_atom)):
            mask = elec_of_atom >= 2 * (i + 1)
            if mask.sum() <= self.mol.n_down - down.sum():
                up = up.at[mask].add(1)
                down = down.at[mask].add(1)

        # distribute remaining electrons such that opposite spin electrons
        # end up close in an attempt to mimic covalent bonds
        nearest_neighbor_indices = jnp.argsort(
            pairwise_distance(self.mol.coords, self.mol.coords)
        )
        center = None
        i = 0
        while (elec_of_atom - up - down).sum():
            if center is None:
                center = argmax_random_choice(rng, elec_of_atom - up - down)
            else:
                ordered = nearest_neighbor_indices[center]
                center = ordered[(elec_of_atom - up - down)[ordered] > 0][0]
            if i % 2 and self.mol.n_down - down.sum() > 0:
                down = down.at[center].add(1)
            else:
                up = up.at[center].add(1)
            i += 1
        assert up.sum() == self.mol.n_up
        assert down.sum() == self.mol.n_down
        return up, down

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
