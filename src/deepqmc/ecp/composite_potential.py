from typing import Optional

import jax
import jax.numpy as jnp

from ..physics import LaplacianFactory, NuclearCoulombPotential, Potential
from ..types import (
    Energy,
    KeyArray,
    ParametrizedWaveFunction,
    Params,
    PhysicalConfiguration,
    WaveFunction,
)
from .gaussian_type_ecp import GaussianTypeECP
from .pseudo_hamiltonian import PseudoHamiltonian


class CompositePotential(Potential):
    r"""Combination of Gaussian-type ECPs and pseudo Hamiltonians in one molecule.

    Each nucleus is treated either with a Gaussian-type ECP, with a pseudo
    Hamiltonian (PH), or with all electrons. The potential consists of

    - the Coulomb attraction to all nuclei with their effective charges,
    - the local core corrections of the ECP and PH nuclei,
    - the non-local term of the ECP nuclei,
    - the kinetic term modified by the PH nuclei, which reduces to
      :math:`-\frac{1}{2}\Delta` away from the PH cores.

    Args:
        gaussian_type_ecp (~deepqmc.ecp.gaussian_type_ecp.GaussianTypeECP): the
            Gaussian-type ECP of the nuclei in its :attr:`ecp_mask`.
        pseudo_hamiltonian (~deepqmc.ecp.pseudo_hamiltonian.PseudoHamiltonian): the
            pseudo Hamiltonian of the nuclei in its :attr:`ecp_mask`, which must not
            overlap with the one of :data:`gaussian_type_ecp`.
    """

    def __init__(
        self,
        gaussian_type_ecp: GaussianTypeECP,
        pseudo_hamiltonian: PseudoHamiltonian,
    ):
        assert not jnp.any(
            gaussian_type_ecp.ecp_mask & pseudo_hamiltonian.ecp_mask
        ), 'A nucleus cannot have both a Gaussian-type ECP and a pseudo Hamiltonian.'
        self.gaussian_type_ecp = gaussian_type_ecp
        self.pseudo_hamiltonian = pseudo_hamiltonian
        self.ecp_mask = gaussian_type_ecp.ecp_mask | pseudo_hamiltonian.ecp_mask
        # each part holds the full nuclear charge for the nuclei outside its mask
        self.ns_valence = jnp.where(
            gaussian_type_ecp.ecp_mask,
            gaussian_type_ecp.ns_valence,
            pseudo_hamiltonian.ns_valence,
        )
        # Coulomb attraction to all nuclei with their effective charges
        self.coulomb = NuclearCoulombPotential(self.ns_valence)

    def local_potential(self, phys_conf: PhysicalConfiguration) -> Energy:
        return (
            self.coulomb.local_potential(phys_conf)
            + self.gaussian_type_ecp.core_potential(phys_conf)
            + self.pseudo_hamiltonian.core_potential(phys_conf)
        )

    def nonloc_potential(
        self,
        rng: Optional[KeyArray],
        phys_conf: PhysicalConfiguration,
        wf: WaveFunction,
    ) -> Energy:
        return self.gaussian_type_ecp.nonloc_potential(rng, phys_conf, wf)

    def grad_nonloc_potential(
        self,
        wf: ParametrizedWaveFunction,
        rng: KeyArray,
        phys_conf: PhysicalConfiguration,
        params: Params,
    ) -> jax.Array:
        return self.gaussian_type_ecp.grad_nonloc_potential(wf, rng, phys_conf, params)

    def kinetic_term(
        self,
        phys_conf: PhysicalConfiguration,
        wf: WaveFunction,
        laplacian_factory: LaplacianFactory,
    ) -> tuple[Energy, jax.Array, jax.Array]:
        return self.pseudo_hamiltonian.kinetic_term(phys_conf, wf, laplacian_factory)
