"""Tests of the angular quadrature grids and the electron cutoff of the non-local ECP."""

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from deepqmc.ecp.gaussian_type_ecp import GaussianTypeECP
from deepqmc.hamil import MolecularHamiltonian
from deepqmc.molecule import Molecule


def test_cutoff_energy_and_force(helpers):
    # Mg (second row) and F (first row), 2 + 7 valence electrons with ccECP
    mol = Molecule(
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.3]]),
        charges=jnp.array([12, 9]),
        charge=0,
        spin=1,
    )
    hamil = MolecularHamiltonian(mol=mol, ecp_type='ccECP', ecp_nonlocal_electrons=5)
    pot = hamil.pot
    assert isinstance(pot, GaussianTypeECP)
    # by_row: octahedron for Mg, tetrahedron for F
    groups = {int(idx[0]): len(q) for q, idx in pot.quadrature_groups}
    assert groups == {0: 6, 1: 4}

    phys_conf = helpers.phys_conf(hamil)
    _wf, params = helpers.create_ansatz(hamil)
    rng = helpers.rng()

    # the cutoff changes the energy compared to using all electrons
    def nonloc(pot):
        return jax.jit(pot.nonloc_potential, static_argnums=2)(
            rng, phys_conf, lambda pc: _wf.apply(params, pc)
        )

    full_pot = MolecularHamiltonian(mol=mol, ecp_type='ccECP').pot
    assert isinstance(full_pot, GaussianTypeECP)
    e_cutoff = nonloc(pot)
    assert jnp.isfinite(e_cutoff) and not jnp.allclose(e_cutoff, nonloc(full_pot))

    # the force is the gradient of the energy estimator with the same rotations, for a
    # wave function whose nuclear coordinates are fixed (the Hellmann-Feynman force
    # differentiates only the operator)
    def wf_fixed_R(params, pc):
        return _wf.apply(params, jdc.replace(pc, R=jax.lax.stop_gradient(phys_conf.R)))

    def energy(R):
        return pot.nonloc_potential(
            rng, jdc.replace(phys_conf, R=R), lambda pc: wf_fixed_R(params, pc)
        )

    force = jax.jit(pot.grad_nonloc_potential, static_argnums=0)(
        wf_fixed_R, rng, phys_conf, params
    )
    expected = jax.jit(jax.grad(energy))(phys_conf.R)
    assert jnp.all(force != 0)
    assert jnp.allclose(force, expected, rtol=1e-4, atol=1e-4 * jnp.abs(expected).max())
