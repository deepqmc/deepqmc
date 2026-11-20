import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import pytest

from deepqmc.ecp.composite_potential import CompositePotential
from deepqmc.ecp.gaussian_type_ecp import GaussianTypeECP
from deepqmc.ecp.pseudo_hamiltonian import PseudoHamiltonian
from deepqmc.molecule import Molecule
from deepqmc.physics import NuclearCoulombPotential, Potential


@pytest.mark.parametrize('ecp_type', [None, 'bfd', 'ccECP'])
@pytest.mark.parametrize('name', ['LiH', 'C'])
class TestPhysics:
    def test_pseudo_potentials(self, helpers, name, ecp_type, ndarrays_regression):
        mol = helpers.mol(name)
        hamil = helpers.hamil(mol, ecp_type)
        phys_conf = helpers.phys_conf(hamil)
        _wf, params = helpers.create_ansatz(hamil)
        wf = lambda phys_conf: _wf.apply(params, phys_conf)
        ndarrays_regression.check(
            {
                'local_potential': hamil.pot.local_potential(phys_conf),
                'nonlocal_potential': (
                    hamil.pot.nonloc_potential(helpers.rng(), phys_conf, wf)
                    if ecp_type
                    else 0
                ),
            },
        )
        # note: nonlocal_potential is not particularly
        # numerically stable, hence the large tolerance


class TestCompositePotential:
    ECP_TYPE = {'Cl': 'PHcc', 'F': 'ccECP'}

    @pytest.fixture
    def hamil(self, helpers):
        mol = Molecule(
            coords=[[0.0, 0.0, 0.0], [3.08, 0.0, 0.0]],
            charges=[17, 9],
            charge=0,
            spin=0,
            unit='bohr',
        )
        return helpers.hamil(mol, self.ECP_TYPE)

    @pytest.fixture
    def wf(self, helpers, hamil):
        ansatz, params = helpers.create_ansatz(hamil)
        return lambda phys_conf: ansatz.apply(params, phys_conf)

    def test_init(self, hamil):
        assert isinstance(hamil.pot, CompositePotential)
        assert jnp.allclose(hamil.pot.ns_valence, jnp.array([7, 7]))
        assert hamil.n_up + hamil.n_down == 14

    def test_overlapping_masks(self, hamil):
        charges = hamil.mol.charges
        with pytest.raises(AssertionError):
            CompositePotential(
                GaussianTypeECP(charges, 'ccECP', jnp.array([True, True])),
                PseudoHamiltonian(charges, 'PHcc', jnp.array([True, False])),
            )

    def test_local_potential(self, helpers, hamil):
        # the Coulomb attraction is counted once, with the combined effective charges
        pot = hamil.pot
        phys_conf = helpers.phys_conf(hamil)
        rs = helpers.phys_conf(hamil, n=8).r

        def local_potential(potential):
            return jax.vmap(
                lambda r: potential.local_potential(jdc.replace(phys_conf, r=r))
            )(rs)

        expected = (
            local_potential(pot.gaussian_type_ecp)
            + local_potential(pot.pseudo_hamiltonian)
            - local_potential(NuclearCoulombPotential(hamil.mol.charges))
        )
        assert jnp.allclose(local_potential(pot), expected)

    def test_nonloc_potential(self, helpers, hamil, wf):
        phys_conf = helpers.phys_conf(hamil)
        V_nl = hamil.pot.nonloc_potential(helpers.rng(), phys_conf, wf)
        assert V_nl != 0
        assert jnp.allclose(
            V_nl,
            hamil.pot.gaussian_type_ecp.nonloc_potential(helpers.rng(), phys_conf, wf),
        )

    def test_kinetic_term_away_from_ph(self, helpers, hamil, wf):
        # outside of the PH core, the kinetic term reduces to -1/2 laplacian
        phys_conf = helpers.phys_conf(hamil)
        r = jnp.array([-6.0, 0.0, 0.0]) + 0.5 * jax.random.normal(
            helpers.rng(), phys_conf.r.shape
        )
        # the PH of Cl vanishes beyond 3.8 bohr
        assert jnp.linalg.norm(r - hamil.mol.coords[0], axis=-1).min() > 3.8
        phys_conf = jdc.replace(phys_conf, r=r)
        E_kin = hamil.pot.kinetic_term(phys_conf, wf, hamil.lap_factory)[0]
        E_kin_ref = Potential.kinetic_term(hamil.pot, phys_conf, wf, hamil.lap_factory)
        assert jnp.allclose(E_kin, E_kin_ref[0])

    def test_local_energy(self, helpers, hamil):
        ansatz, params = helpers.create_ansatz(hamil)
        phys_conf = helpers.phys_conf(hamil)
        E_loc, _ = hamil.local_energy(ansatz.apply)(helpers.rng(), params, phys_conf)
        assert jnp.isfinite(E_loc)
