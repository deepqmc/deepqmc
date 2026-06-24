from collections.abc import Callable
from functools import partial
from itertools import count
from typing import Any, Optional, Protocol

import jax
import jax.numpy as jnp

from .ecp.gaussian_type_ecp import GaussianTypeECP
from .ecp.pseudo_hamiltonian import PseudoHamiltonian
from .molecule import Molecule
from .physics import (
    LaplacianFactory,
    NuclearCoulombPotential,
    Potential,
    electronic_potential,
    nuclear_energy,
    reverse_forward_laplacian,
)
from .types import (
    Energy,
    KeyArray,
    ParametrizedWaveFunction,
    Params,
    PhysicalConfiguration,
    Stats,
)

__all__ = ['MolecularHamiltonian']


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
            ansatz (~deepqmc.types.ParametrizedWaveFunction): the wave function ansatz.

        Returns:
            :class:`~collections.abc.Callable`\[[:data:`~deepqmc.types.KeyArray` |
            None, :data:`~deepqmc.types.Params`,
            :class:`~deepqmc.types.PhysicalConfiguration`],
            tuple[:data:`~deepqmc.types.Energy`, :data:`~deepqmc.types.Stats`]\]:
            a function that evaluates the local energy of :data:`ansatz` at a given
            physical configuration.
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
        ecp_type (str): If set, use the appropriate pseudopotential or effective core
            potential (ECP). The string is passed to :func:`pyscf.gto.M()` as
            :data:`'ecp'` argument.
            Supports ECPs that are implemented in the pyscf package, e.g. :data:`'bfd'`
            [Burkatzki et al. 2007] or :data:`'ccECP'` [Bennett et al. 2017].
            Supports PseudoHamiltonians from [Ichibha23] and [Fu25], e.g. :data:`'PHcc'`
            or :data:`'PHhf'`.
        ecp_mask (list[bool]): list of True and False values (:math:`N_\text{nuc}`)
            specifying whether to use an ECP for each nucleus.
        elec_std (float): optional, a default value of the scaling factor
            of the spread of electrons around the nuclei.
        laplacian_factory (~deepqmc.physics.LaplacianFactory): creates a function that
            returns a tuple containing the laplacian and gradient of the wave function.
    """

    mol: Molecule
    ecp_type: Optional[str]
    ecp_mask: jax.Array
    elec_std: float
    lap_factory: LaplacianFactory
    pot: Potential
    n_nuc: int
    n_up: int
    n_down: int
    ns_valence: jax.Array

    def __init__(
        self,
        *,
        mol: Molecule,
        ecp_type: Optional[str] = None,
        ecp_mask: Optional[list[bool]] = None,
        elec_std: float = 1.0,
        laplacian_factory: LaplacianFactory = reverse_forward_laplacian,
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

        self.lap_factory = laplacian_factory
        if self.ecp_mask.any():
            assert (
                self.ecp_type is not None
            ), 'ECP type must be specified if ECPs are used.'
            if 'PH' in str(self.ecp_type):
                self.pot = PseudoHamiltonian(mol.charges, self.ecp_type, self.ecp_mask)
            else:
                self.pot = GaussianTypeECP(mol.charges, self.ecp_type, self.ecp_mask)
        else:
            self.pot = NuclearCoulombPotential(mol.charges)

        n_elec = int(sum(self.pot.ns_valence) - mol.charge)
        assert not (n_elec + mol.spin) % 2
        assert n_elec > 1, 'The system must contain at least two active electrons.'

        self.n_nuc = len(mol.charges)
        self.n_up = (n_elec + mol.spin) // 2
        self.n_down = (n_elec - mol.spin) // 2
        self.ns_valence = self.pot.ns_valence

        self.mol_shells = [get_shell(z) for z in self.mol.charges]
        self.mol_ecp_shells = [
            get_shell(z + 1) - 1 for z in self.mol.charges - self.ns_valence
        ]

    def local_energy(self, ansatz: ParametrizedWaveFunction) -> Callable[
        [Optional[KeyArray], Params, PhysicalConfiguration],
        tuple[Energy, Stats],
    ]:
        def loc_ene(
            rng: Optional[KeyArray], params: Params, phys_conf: PhysicalConfiguration
        ) -> tuple[Energy, Stats]:
            wf = partial(ansatz, params)

            Es_kin, lap_log_psis, quantum_force_2_sum = self.pot.kinetic_term(
                phys_conf, wf, self.lap_factory
            )
            Es_nuc = nuclear_energy(phys_conf, self.ns_valence)
            Vs_el = electronic_potential(phys_conf)
            Vs_loc = self.pot.local_potential(phys_conf)
            Vs_nl = self.pot.nonloc_potential(rng, phys_conf, wf)
            Es_loc = Es_kin + Vs_loc + Vs_nl + Vs_el + Es_nuc
            stats = {
                'hamil/V_el': Vs_el,
                'hamil/E_kin': Es_kin,
                'hamil/V_loc': Vs_loc,
                'hamil/V_nl': Vs_nl,
                'hamil/lap': lap_log_psis,
                'hamil/quantum_force': quantum_force_2_sum,
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
