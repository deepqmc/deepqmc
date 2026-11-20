import logging
from collections.abc import Callable, Mapping
from functools import partial
from itertools import count
from typing import Any, Optional, Protocol, Union

import jax
import jax.numpy as jnp
from pyscf.lib.parameters import ELEMENTS

from .ecp.composite_potential import CompositePotential
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

log = logging.getLogger(__name__)


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


def per_nucleus_ecp_types(
    charges: jax.Array,
    ecp_type: Optional[Union[str, Mapping[Union[str, int], str]]],
    ecp_mask: Optional[list[bool]],
) -> list[Optional[str]]:
    r"""Return the ECP type of each nucleus, :data:`None` for all-electron nuclei.

    Args:
        charges (~jax.Array): the atomic numbers of the nuclei.
        ecp_type (str | ~collections.abc.Mapping[str | int, str] | None): a single
            ECP type, or a mapping from element symbols or atomic numbers to ECP
            types, see :class:`MolecularHamiltonian`.
        ecp_mask (list[bool] | None): whether to use an ECP for each nucleus, only
            used if :data:`ecp_type` is a string. Defaults to the atoms larger than
            He.
    """
    if ecp_type is None:
        return [None] * len(charges)
    if isinstance(ecp_type, Mapping):
        assert ecp_mask is None, (
            "'ecp_mask' cannot be used if 'ecp_type' is a mapping, the mapping"
            ' selects the nuclei with an ECP.'
        )
        types_by_charge: dict[int, str] = {}
        for element, element_ecp_type in ecp_type.items():
            charge = element if isinstance(element, int) else ELEMENTS.index(element)
            assert (
                charge not in types_by_charge
            ), f'ECP type of {ELEMENTS[charge]} specified more than once.'
            types_by_charge[charge] = element_ecp_type
        return [types_by_charge.get(int(c)) for c in charges]
    if ecp_mask is None:
        # use ECP only for atoms larger than He
        ecp_mask = list(charges > 2)
    assert len(ecp_mask) == len(charges), "Incompatible shape of 'ecp_mask'!"
    return [ecp_type if m else None for m in ecp_mask]


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
            ~collections.abc.Callable[[~deepqmc.types.KeyArray | None, ~deepqmc.types.Params, ~deepqmc.types.PhysicalConfiguration], tuple[~deepqmc.types.Energy, ~deepqmc.types.Stats]]: a function that evaluates the local energy of ``ansatz``
                at a given physical configuration.
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
        ecp_type (str | ~collections.abc.Mapping[str | int, str]): If set, use the
            appropriate pseudopotential or effective core potential (ECP). The string
            is passed to :func:`pyscf.gto.M()` as :data:`'ecp'` argument.
            Supports ECPs that are implemented in the pyscf package, e.g. :data:`'bfd'`
            [Burkatzki et al. 2007] or :data:`'ccECP'` [Bennett et al. 2017].
            Supports PseudoHamiltonians from [Ichibha23] and [Fu26], e.g. :data:`'PHcc'`
            or :data:`'PHhf'`.
            A mapping from element symbols or atomic numbers to ECP types, e.g.
            :data:`{'Fe': 'PHcc', 'O': 'ccECP'}`, selects the ECP per element.
            Elements not in the mapping are treated with all electrons, and
            Gaussian-type ECPs and PseudoHamiltonians can be combined this way.
            At most one Gaussian-type ECP and one PseudoHamiltonian type can be used.
        ecp_mask (list[bool]): list of True and False values (:math:`N_\text{nuc}`)
            specifying whether to use an ECP for each nucleus. Only used if
            :data:`ecp_type` is a string.
        elec_std (float): optional, a default value of the scaling factor
            of the spread of electrons around the nuclei.
        laplacian_factory (~deepqmc.physics.LaplacianFactory): creates a function that
            returns a tuple containing the laplacian and gradient of the wave function.
        ecp_nonlocal_electrons (int | None): optional, if given, only this many
            electrons closest to each nucleus contribute to the non-local part of
            Gaussian-type ECPs. If :data:`None` (default), all electrons contribute.
        ecp_quadrature (str): the angular quadrature grid of the non-local part of
            Gaussian-type ECPs, ``'by_row'`` (default; 4 points for first-row,
            6 points for second-row and 12 points for heavier elements),
            ``'tetrahedron'`` (4 points), ``'octahedron'`` (6 points) or
            ``'icosahedron'`` (12 points). Using fewer grid points increases the
            variance, but is cheaper.
    """

    mol: Molecule
    ecp_type: Optional[Union[str, Mapping[Union[str, int], str]]]
    ecp_types: list[Optional[str]]
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
        ecp_type: Optional[Union[str, Mapping[Union[str, int], str]]] = None,
        ecp_mask: Optional[list[bool]] = None,
        elec_std: float = 1.0,
        laplacian_factory: LaplacianFactory = reverse_forward_laplacian,
        ecp_nonlocal_electrons: Optional[int] = None,
        ecp_quadrature: str = 'by_row',
    ):
        self.mol = mol
        self.elec_std = elec_std
        self.ecp_type = ecp_type
        self.ecp_types = per_nucleus_ecp_types(mol.charges, ecp_type, ecp_mask)

        ph_mask = jnp.array([t is not None and 'PH' in str(t) for t in self.ecp_types])
        self.ecp_mask = jnp.array([t is not None for t in self.ecp_types])
        gto_mask = self.ecp_mask & ~ph_mask
        ph_types = {t for t, m in zip(self.ecp_types, ph_mask) if m}
        gto_types = {t for t, m in zip(self.ecp_types, gto_mask) if m}
        assert len(ph_types) <= 1, f'Only one PseudoHamiltonian type, got {ph_types}.'
        assert len(gto_types) <= 1, f'Only one Gaussian-type ECP, got {gto_types}.'

        self.lap_factory = laplacian_factory
        if ph_mask.any() and not gto_mask.any() and ecp_nonlocal_electrons is not None:
            log.warning(
                'ecp_nonlocal_electrons is ignored for PseudoHamiltonians'
                f' ({ph_types}), which have no non-local term.'
            )
        gaussian_type_ecp = (
            GaussianTypeECP(
                mol.charges,
                gto_types.pop(),
                gto_mask,
                ecp_nonlocal_electrons,
                ecp_quadrature,
            )
            if gto_mask.any()
            else None
        )
        pseudo_hamiltonian = (
            PseudoHamiltonian(mol.charges, ph_types.pop(), ph_mask)
            if ph_mask.any()
            else None
        )
        if gaussian_type_ecp is not None and pseudo_hamiltonian is not None:
            self.pot = CompositePotential(gaussian_type_ecp, pseudo_hamiltonian)
        elif gaussian_type_ecp is not None:
            self.pot = gaussian_type_ecp
        elif pseudo_hamiltonian is not None:
            self.pot = pseudo_hamiltonian
        else:
            self.pot = NuclearCoulombPotential(mol.charges)

        n_elec = int(sum(self.pot.ns_valence) - mol.charge)
        assert not (n_elec + mol.spin) % 2
        assert n_elec > 1, 'The system must contain at least two active electrons.'
        assert ecp_nonlocal_electrons is None or ecp_nonlocal_electrons <= n_elec, (
            f'ecp_nonlocal_electrons ({ecp_nonlocal_electrons}) must not exceed the'
            f' number of valence electrons ({n_elec}).'
        )

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
            coords (~jax.Array): optional, nuclear coordinates (:math:`N_\text{nuc}`, 3).
        """
        coords = coords if coords is not None else self.mol.coords
        pyscf_kwargs = {
            'atom': [(int(c), r.tolist()) for c, r in zip(self.mol.charges, coords)],
            'charge': self.mol.charge,
            'spin': self.mol.spin,
            'ecp': {int(c): t for c, t in zip(self.mol.charges, self.ecp_types) if t},
            'unit': 'bohr',
        }
        return pyscf_kwargs
