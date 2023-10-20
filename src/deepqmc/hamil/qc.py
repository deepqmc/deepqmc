from itertools import count

import jax.numpy as jnp
import jax_dataclasses as jdc
from jax import lax, random, vmap

from ..physics import (
    NuclearCoulombPotential,
    electronic_potential,
    laplacian,
    nuclear_energy,
    pairwise_distance,
)
from ..pp.ecp_potential import EcpTypePseudopotential
from ..types import PhysicalConfiguration
from ..utils import argmax_random_choice
from .base import Hamiltonian

__all__ = ['MolecularHamiltonian']


def get_shell(z):
    # returns the number of (at least partially) occupied shells for 'z' electrons
    # 'get_shell(z+1)-1' yields the number of fully occupied shells for 'z' electrons
    max_elec = 0
    for n in count():
        if z <= max_elec:
            break
        max_elec += 2 * (1 + n) ** 2
    return n


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
        pp_type (str): If set, use the appropriate pseudopotential. The string is passed
            to :func:`pyscf.gto.M()` as :data:`'ecp'` argument. Supports
            pseudopotentials that are implemented in the pyscf package, e.g.
            :data:`'bfd'` [Burkatzki et al. 2007] or :data:`'ccECP'`
            [Bennett et al. 2017].
        pp_mask (list, (:math:`N_\text{nuc}`)): list of True and False values specifying
            whether to use a pseudopotential for each nucleus
        elec_std (float): optional, a default value of the scaling factor
        of the spread of electrons around the nuclei.
    """

    def __init__(self, *, mol, pp_type=None, pp_mask=None, elec_std=1.0):
        self.mol = mol
        self.elec_std = elec_std
        self.pp_type = pp_type

        if pp_type is None:
            pp_mask = [False] * len(mol.charges)
        elif pp_mask is None:
            # use PP only for atoms larger than He
            pp_mask = mol.charges > 2
        assert len(pp_mask) == len(mol.charges), "Incompatible shape of 'pp_mask'!"

        self.pp_mask = jnp.array(pp_mask)

        # Derived properties
        if self.pp_type is None:
            self.potential = NuclearCoulombPotential(mol.charges)
        else:
            self.potential = EcpTypePseudopotential(mol.charges, pp_type, self.pp_mask)

        n_elec = int(sum(self.potential.ns_valence) - mol.charge)
        assert not (n_elec + mol.spin) % 2
        assert n_elec > 1, 'The system must contain at least two active electrons.'

        self.n_nuc = len(mol.charges)
        self.n_up = (n_elec + mol.spin) // 2
        self.n_down = (n_elec - mol.spin) // 2
        self.ns_valence = self.potential.ns_valence

        self.mol_shells = [get_shell(z) for z in self.mol.charges]
        self.mol_pp_shells = [
            get_shell(z + 1) - 1 for z in self.mol.charges - self.ns_valence
        ]

    def init_sample(self, rng, R, n, elec_std=None):
        r"""
        Guess some initial electron positions.

        Tries to make an educated guess about plausible initial electron
        configurations. Places electrons according to normal distributions
        centered on the nuclei. If the molecule is not neutral, extra electrons
        are placed on or removed from random nuclei. The resulting configurations
        are usually very crude, a subsequent, thorough equilibration is needed.

        Args:
            rng (jax.random.PRNGKey): key used for PRNG.
            R (float, (:math:`N_\text{nuc}`, 3)): nuclear coordinates of a single
                molecular geometry
            n (int): the number of configurations to generate.
                electrons around the nuclei.
        """
        assert R.ndim == 2

        Rs = jnp.tile(R[None], (n, 1, 1))
        return vmap(self.init_single_sample, (0, 0, None))(
            random.split(rng, n), Rs, elec_std
        )

    def init_single_sample(self, rng, R, elec_std):
        rng_remainder, rng_normal, rng_spin = random.split(rng, 3)
        valence_electrons = self.potential.ns_valence - self.mol.charge / self.n_nuc
        electrons_of_atom = jnp.floor(valence_electrons).astype(jnp.int32)

        def cond_fn(value):
            _, electrons_of_atom = value
            return (
                self.potential.ns_valence.sum()
                - self.mol.charge
                - electrons_of_atom.sum()
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
        up = (jnp.cumsum(up)[:, None] <= jnp.arange(self.n_up)).sum(axis=0)
        down = (jnp.cumsum(down)[:, None] <= jnp.arange(self.n_down)).sum(axis=0)
        idxs = jnp.concatenate([up, down])
        centers = R[idxs]
        std = (elec_std or self.elec_std) * jnp.sqrt(self.mol.charges)[idxs][..., None]
        r = centers + std * random.normal(rng_normal, centers.shape)
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
            increment = jnp.where(mask & (mask.sum() + down.sum() <= self.n_down), 1, 0)
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
            is_down = (i % 2) & (down.sum() < self.n_down)
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
            Es_nuc = nuclear_energy(phys_conf, self.ns_valence)
            Vs_el = electronic_potential(phys_conf)
            Vs_loc = self.potential.local_potential(phys_conf)
            Vs_nl = self.potential.nonloc_potential(rng, phys_conf, wf)
            Es_loc = Es_kin + Vs_loc + Vs_nl + Vs_el + Es_nuc
            stats = {
                'hamil/V_el': Vs_el,
                'hamil/E_kin': Es_kin,
                'hamil/V_loc': Vs_loc,
                'hamil/V_nl': Vs_nl,
                'hamil/lap': lap_log_psis,
                'hamil/quantum_force': (quantum_force**2).sum(axis=-1),
            }

            result = (Es_loc, quantum_force) if return_grad else Es_loc
            return result, stats

        return loc_ene

    def as_pyscf(self, coords):
        r"""Return nuclear charges and coordinates in a format pyscf can parse.

        Args:
            coords (jax.Array): nuclear coordinates, shape [n_nuc, 3].
        """
        return [(int(charge), coord) for coord, charge in zip(coords, self.mol.charges)]
