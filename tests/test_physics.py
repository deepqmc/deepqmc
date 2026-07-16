import jax.numpy as jnp

from deepqmc.physics import electronic_potential, nuclear_energy
from deepqmc.types import PhysicalConfiguration


class TestCoulombTerms:
    def test_nuclear_energy_and_electronic_potential(self):
        phys_conf = PhysicalConfiguration(
            R=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
            r=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            mol_idx=jnp.array(0),
        )
        ns_valence = jnp.array([1.0, 1.0])

        assert jnp.allclose(nuclear_energy(phys_conf, ns_valence), 1 / 1.4)
        assert jnp.allclose(electronic_potential(phys_conf), 1 / 1.0)
