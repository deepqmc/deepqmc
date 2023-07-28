import jax.numpy as jnp
import jax_dataclasses as jdc
from jax import lax, random, vmap

from ..physics import (
    electronic_potential,
    laplacian,
    local_potential,
    nonlocal_potential,
    nuclear_energy,
    pairwise_distance,
)
from ..types import PhysicalConfiguration
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
        elec_std (float): optional, a default value of the scaling factor
        of the spread of electrons around the nuclei.
    """

    def __init__(self, *, mol, elec_std=1.0):
        self.mol = mol
        self.elec_std = elec_std

    def init_sample(self, rng, Rs, n, elec_std=None):
        r"""
        Guess some initial electron positions.

        Tries to make an educated guess about plausible initial electron
        configurations. Places electrons according to normal distributions
        centered on the nuclei. If the molecule is not neutral, extra electrons
        are placed on or removed from random nuclei. The resulting configurations
        are usually very crude, a subsequent, thorough equilibration is needed.

        Args:
            rng (jax.random.PRNGKey): key used for PRNG.
            Rs (float, (:data:`n`, :math:`N_\text{nuc}`, 3)): nuclear coordinates,
                it is broadcasted if batch dimension (:data:`n`) is omitted
            n (int): the number of configurations to generate.
                electrons around the nuclei.
        """

        Rs = jnp.tile(Rs[None], (n, 1, 1)) if Rs.ndim == 2 else Rs
        return vmap(self.init_single_sample)(random.split(rng, n), Rs)

    def init_single_sample(self, rng, R):
        rng_remainder, rng_normal, rng_spin = random.split(rng, 3)
        valence_electrons = self.mol.ns_valence - self.mol.charge / self.mol.n_nuc
        electrons_of_atom = jnp.floor(valence_electrons).astype(jnp.int32)

        def cond_fn(value):
            _, electrons_of_atom = value
            return (
                self.mol.ns_valence.sum() - self.mol.charge - electrons_of_atom.sum()
                > 0
            )

        def body_fn(value):
            rng, electrons_of_atom = value
            rng, rng_categorical = random.split(rng)
            atom_idx = random.categorical(
                rng_categorical, valence_electrons - electrons_of_atom, shape=()
            )
            electrons_of_atom = electrons_of_atom.at[atom_idx].add(1)
            return rng, electrons_of_atom

        _, electrons_of_atom = lax.while_loop(
            cond_fn, body_fn, (rng_remainder, electrons_of_atom)
        )
        rng_spin = random.split(rng_spin, 1)
        up, down = self.distribute_spins(rng_spin, R, electrons_of_atom)
        up = (jnp.cumsum(up)[:, None] <= jnp.arange(self.mol.n_up)).sum(axis=0)
        down = (jnp.cumsum(down)[:, None] <= jnp.arange(self.mol.n_down)).sum(axis=0)
        idxs = jnp.concatenate([up, down])
        centers = R[idxs]
        r = centers + random.normal(rng_normal, centers.shape)
        return PhysicalConfiguration(R, r, jnp.array(0))

    def distribute_spins(self, rng, R, elec_of_atom):
        up, down = jnp.zeros_like(elec_of_atom), jnp.zeros_like(elec_of_atom)
        # try to distribute electron pairs evenly across atoms

        def pair_cond_fn(value):
            i, *_ = value
            return i < jnp.max(elec_of_atom)

        def pair_body_fn(value):
            i, up, down = value
            mask = elec_of_atom >= 2 * (i + 1)
            increment = jnp.where(
                mask & (mask.sum() + down.sum() <= self.mol.n_down), 1, 0
            )
            up = up + increment
            down = down + increment
            return i + 1, up, down

        _, up, down = lax.while_loop(pair_cond_fn, pair_body_fn, (0, up, down))

        # distribute remaining electrons such that opposite spin electrons
        # end up close in an attempt to mimic covalent bonds
        dists = pairwise_distance(R, R).at[jnp.diag_indices(len(R))].set(jnp.inf)
        nearest_neighbor_indices = jnp.argsort(dists)

        def spin_cond_fn(value):
            _, _, up, down = value
            return (up + down < elec_of_atom).any()

        def spin_body_fn(value):
            i, center, up, down = value
            is_down = (i % 2) & (down.sum() < self.mol.n_down)
            up = up.at[center].add(1 - is_down)
            down = down.at[center].add(is_down)
            ordering = nearest_neighbor_indices[center]
            ordered_has_remainder = (elec_of_atom - up - down)[ordering] > 0
            first_ordered_has_remainder = jnp.argmax(ordered_has_remainder)
            center = ordering[first_ordered_has_remainder]
            return i + 1, center, up, down

        center = argmax_random_choice(rng, elec_of_atom - up - down)
        *_, up, down = lax.while_loop(
            spin_cond_fn, spin_body_fn, (jnp.array(0), center, up, down)
        )

        return up, down

    def local_energy(self, wf, return_grad=False):
        def loc_ene(rng, phys_conf):
            def wave_function(r):
                pc = jdc.replace(phys_conf, r=r.reshape(-1, 3))
                return wf(pc).log

            lap_log_psis, quantum_force = laplacian(wave_function)(
                phys_conf.r.flatten()
            )
            Es_kin = -0.5 * (lap_log_psis + (quantum_force**2).sum(axis=-1))
            Es_nuc = nuclear_energy(phys_conf, self.mol)
            Vs_el = electronic_potential(phys_conf)
            Vs_loc = local_potential(phys_conf, self.mol)
            Es_loc = Es_kin + Vs_loc + Vs_el + Es_nuc
            stats = {
                'hamil/V_el': Vs_el,
                'hamil/E_kin': Es_kin,
                'hamil/V_loc': Vs_loc,
                'hamil/lap': lap_log_psis,
                'hamil/quantum_force': (quantum_force**2).sum(axis=-1),
            }
            if self.mol.any_pp:
                Vs_nl = nonlocal_potential(rng, phys_conf, self.mol, wf)
                Es_loc += Vs_nl
                stats = {**stats, 'hamil/V_nl': Vs_nl}

            result = (Es_loc, quantum_force) if return_grad else Es_loc
            return result, stats

        return loc_ene
