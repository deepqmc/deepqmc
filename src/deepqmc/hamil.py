from collections.abc import Callable
from functools import partial
from itertools import count
from typing import Any, Optional, Protocol

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from .ecp.gaussian_type_ecp import GaussianTypeECP
from .molecule import Molecule
from .physics import (
    NuclearCoulombPotential,
    electronic_potential,
    laplacian,
    nuclear_energy,
    pairwise_distance,
)
from .types import (
    Energy,
    KeyArray,
    ParametrizedWaveFunction,
    Params,
    PhysicalConfiguration,
    Stats,
)
from .utils import argmax_random_choice

__all__ = ['MolecularHamiltonian', 'LaplacianFactory']


class LaplacianFactory(Protocol):
    r"""Protocol class for Laplacian factories.

    A Laplacian factory takes as input a function and returns a function that
    computes the laplacian and gradient of the input function
    """

    def __call__(
        self, f: Callable[[jax.Array], jax.Array]
    ) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]: ...


def get_shell(z):
    # returns the number of (at least partially) occupied shells for 'z' electrons
    # 'get_shell(z+1)-1' yields the number of fully occupied shells for 'z' electrons
    max_elec = 0
    n = 0
    for n in count():
        if z <= max_elec:
            break
        max_elec += 2 * (1 + n) ** 2
    return n


class Hamiltonian(Protocol):
    r"""Protocol for :class:`~deepqmc.types.Hamiltonian` objects.

    :class:`~deepqmc.types.Hamiltonian` objects represent the Hamiltonian of the system
    under investigation. New Hamiltonians should implement this protocol to be
    compatible with the DeepQMC software suite. The :class:`~deepqmc.types.Hamiltonian`
    object holds information about the system and implements the local energy factory.
    """

    def local_energy(self, ansatz: ParametrizedWaveFunction) -> Callable[
        [Optional[KeyArray], Params, PhysicalConfiguration],
        tuple[Energy, Stats],
    ]:
        r"""
        Return a function that calculates the local energy of the wave function.

        Args:
            wf (~deepqmc.types.ParametrizedWaveFunction): the wave function ansatz.
            return_grad (bool): whether to return a tuple with the quantum force.

        Returns:
            :class:`Callable[r, ...]`: a function that evaluates
            the local energy of :data:`wf` at :data:`r`.
        """
        ...


class MolecularHamiltonian(Hamiltonian):
    r"""
    Hamiltonian of non-relativistic molecular systems.

    The system consists of nuclei with fixed positions and electrons moving
    around them. The total energy is defined as the sum of the nuclear-nuclear
    and electron-electron repulsion, the nuclear-electron attraction, and the
    kinetic energy of the electrons:
    :math:`E=V_\text{nuc-nuc} + V_\text{el-el} + V_\text{nuc-el} + E_\text{kin}`.

    Args:
        mol (~deepqmc.molecule.Molecule): the molecule to consider
        ecp_type (str): If set, use the appropriate effective core potential (ECP). The
            string is passed to :func:`pyscf.gto.M()` as :data:`'ecp'` argument.
            Supports ECPs that are implemented in the pyscf package, e.g. :data:`'bfd'`
            [Burkatzki et al. 2007] or :data:`'ccECP'` [Bennett et al. 2017].
        ecp_mask (list[bool]): list of True and False values (:math:`N_\text{nuc}`)
            specifying whether to use an ECP for each nucleus.
        elec_std (float): optional, a default value of the scaling factor
            of the spread of electrons around the nuclei.
        laplacian_factory (~deepqmc.hamil.LaplacianFactory): creates a function that
            returns a tuple containing the laplacian and gradient of the wave function.
    """

    def __init__(
        self,
        *,
        mol: Molecule,
        ecp_type: Optional[str] = None,
        ecp_mask: Optional[list[bool]] = None,
        elec_std: float = 1.0,
        laplacian_factory: LaplacianFactory = laplacian,
    ):
        self.mol = mol
        self.elec_std = elec_std
        self.ecp_type = ecp_type

        if ecp_type is None:
            ecp_mask = [False] * len(mol.charges)
        elif ecp_mask is None:
            # use ECP only for atoms larger than He
            ecp_mask = list(mol.charges > 2)
        assert len(ecp_mask) == len(mol.charges), "Incompatible shape of 'ecp_mask'!"

        self.ecp_mask = jnp.array(ecp_mask)

        self.laplacian = laplacian_factory
        self.potential: (
            NuclearCoulombPotential | GaussianTypeECP
        )  # mypy otherwise complains about the following assignment
        if self.ecp_mask.any():
            self.potential = GaussianTypeECP(mol.charges, ecp_type, self.ecp_mask)
        else:
            self.potential = NuclearCoulombPotential(mol.charges)

        n_elec = int(sum(self.potential.ns_valence) - mol.charge)
        assert not (n_elec + mol.spin) % 2
        assert n_elec > 1, 'The system must contain at least two active electrons.'

        self.n_nuc = len(mol.charges)
        self.n_up = (n_elec + mol.spin) // 2
        self.n_down = (n_elec - mol.spin) // 2
        self.ns_valence = self.potential.ns_valence

        self.mol_shells = [get_shell(z) for z in self.mol.charges]
        self.mol_ecp_shells = [
            get_shell(z + 1) - 1 for z in self.mol.charges - self.ns_valence
        ]

    def init_sample(
        self, rng: KeyArray, R: jax.Array, n: int, elec_std: Optional[float] = None
    ) -> PhysicalConfiguration:
        r"""
        Guess some initial electron positions.

        Tries to make an educated guess about plausible initial electron
        configurations. Places electrons according to normal distributions
        centered on the nuclei. If the molecule is not neutral, extra electrons
        are placed on or removed from random nuclei. The resulting configurations
        are usually very crude, a subsequent, thorough equilibration is needed.

        Args:
            rng (~deepqmc.types.KeyArray): key used for PRNG.
            R (jax.Array): nuclear coordinates of a single molecular geometry
                (:math:`N_\text{nuc}`, 3)
            n (int): the number of initial electron configurations to generate.

        Returns:
            :class:`~deepqmc.types.PhysicalConfiguration`:
            initial electron and nuclei
                configurations
        """
        assert R.ndim == 2

        Rs = jnp.tile(R[None], (n, 1, 1))
        return jax.vmap(self.init_single_sample, (0, 0, None))(
            jax.random.split(rng, n), Rs, elec_std
        )

    def init_single_sample(
        self, rng: KeyArray, R: jax.Array, elec_std: Optional[float]
    ) -> PhysicalConfiguration:
        rng_remainder, rng_normal, rng_spin = jax.random.split(rng, 3)
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
            rng, rng_categorical = jax.random.split(rng)
            atom_idx = jax.random.categorical(
                rng_categorical, valence_electrons - electrons_of_atom, shape=()
            )
            electrons_of_atom = electrons_of_atom.at[atom_idx].add(1)
            return rng, electrons_of_atom

        _, electrons_of_atom = jax.lax.while_loop(
            cond_fn, body_fn, (rng_remainder, electrons_of_atom)
        )
        up, down = self.distribute_spins(rng_spin, R, electrons_of_atom)
        up = (jnp.cumsum(up)[:, None] <= jnp.arange(self.n_up)).sum(axis=0)
        down = (jnp.cumsum(down)[:, None] <= jnp.arange(self.n_down)).sum(axis=0)
        idxs = jnp.concatenate([up, down])
        centers = R[idxs]
        std = (elec_std or self.elec_std) * jnp.sqrt(self.mol.charges)[idxs][..., None]
        r = centers + std * jax.random.normal(rng_normal, centers.shape)
        return PhysicalConfiguration(R, r, jnp.array(0))  # type: ignore

    def distribute_spins(
        self, rng: KeyArray, R: jax.Array, elec_of_atom: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
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

        _, up, down = jax.lax.while_loop(pair_cond_fn, pair_body_fn, (0, up, down))

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
        *_, up, down = jax.lax.while_loop(
            spin_cond_fn, spin_body_fn, (jnp.array(0), center, up, down)
        )

        return up, down

    def local_energy(self, ansatz: ParametrizedWaveFunction) -> Callable[
        [Optional[KeyArray], Params, PhysicalConfiguration],
        tuple[Energy, Stats],
    ]:
        def loc_ene(
            rng: Optional[KeyArray], params: Params, phys_conf: PhysicalConfiguration
        ) -> tuple[Energy, Stats]:
            wf = partial(ansatz, params)

            def wave_function(r: jax.Array) -> jax.Array:
                pc = jdc.replace(phys_conf, r=r.reshape(-1, 3))
                return wf(pc).log

            lap_log_psis, quantum_force = self.laplacian(wave_function)(
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

            return Es_loc, stats

        return loc_ene

    def as_pyscf(self, *, coords: Optional[jax.Array] = None) -> dict[str, Any]:
        r"""Return the hamiltonian parameters in format pyscf can parse.

        Args:
            coords (jax.Array): optional, nuclear coordinates (:math:`N_\text{nuc}`, 3).
        """
        coords = coords if coords is not None else self.mol.coords
        pyscf_kwargs = {
            'atom': [(int(c), r.tolist()) for c, r in zip(self.mol.charges, coords)],
            'charge': self.mol.charge,
            'spin': self.mol.spin,
            'ecp': {int(c): self.ecp_type for c in self.mol.charges[self.ecp_mask]},
            'unit': 'bohr',
        }
        return pyscf_kwargs
