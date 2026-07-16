import jax.numpy as jnp

from deepqmc.loss.energy import (
    compute_local_energy,
    compute_mean_energy,
    compute_mean_energy_tangent,
)
from deepqmc.parallel import pmap
from deepqmc.utils import tree_stack

EXPECTED_STAT_KEYS = {
    'hamil/V_el',
    'hamil/E_kin',
    'hamil/V_loc',
    'hamil/V_nl',
    'hamil/lap',
    'hamil/quantum_force',
}


class TestComputeLocalEnergy:
    def test_shape_and_stats(self, helpers):
        hamil = helpers.hamil()
        ansatz, params = helpers.create_ansatz(hamil)
        phys_conf, weight = helpers.batch_phys_conf(
            hamil, mol_batch=2, n_states=1, elec_batch=3
        )
        stacked_params = tree_stack([params])
        local_energy, stats = compute_local_energy(
            helpers.rng(1), hamil, ansatz.apply, stacked_params, phys_conf
        )
        assert local_energy.shape == phys_conf.batch_shape == (2, 1, 3)
        assert set(stats.keys()) == EXPECTED_STAT_KEYS
        for value in stats.values():
            assert value.shape == (2, 1)
        assert jnp.all(jnp.isfinite(local_energy))
        for value in stats.values():
            assert jnp.all(jnp.isfinite(value))

    def test_determinism(self, helpers):
        hamil = helpers.hamil()
        ansatz, params = helpers.create_ansatz(hamil)
        phys_conf, weight = helpers.batch_phys_conf(
            hamil, mol_batch=2, n_states=1, elec_batch=3
        )
        stacked_params = tree_stack([params])
        local_energy_1, stats_1 = compute_local_energy(
            helpers.rng(1), hamil, ansatz.apply, stacked_params, phys_conf
        )
        local_energy_2, stats_2 = compute_local_energy(
            helpers.rng(1), hamil, ansatz.apply, stacked_params, phys_conf
        )
        assert jnp.array_equal(local_energy_1, local_energy_2)
        assert helpers.pytree_allclose(stats_1, stats_2)

    def test_rng_does_not_affect_local_energy_without_ecp(self, helpers):
        hamil = helpers.hamil(ecp_type=None)
        ansatz, params = helpers.create_ansatz(hamil)
        phys_conf, weight = helpers.batch_phys_conf(
            hamil, mol_batch=2, n_states=1, elec_batch=3
        )
        stacked_params = tree_stack([params])
        local_energy_1, _ = compute_local_energy(
            helpers.rng(1), hamil, ansatz.apply, stacked_params, phys_conf
        )
        local_energy_2, _ = compute_local_energy(
            helpers.rng(2), hamil, ansatz.apply, stacked_params, phys_conf
        )
        assert jnp.array_equal(local_energy_1, local_energy_2)

    def test_batch_size_matches_unbatched(self, helpers):
        hamil = helpers.hamil()
        ansatz, params = helpers.create_ansatz(hamil)
        phys_conf, weight = helpers.batch_phys_conf(
            hamil, mol_batch=2, n_states=1, elec_batch=4
        )
        stacked_params = tree_stack([params])
        local_energy_full, _ = compute_local_energy(
            helpers.rng(1), hamil, ansatz.apply, stacked_params, phys_conf
        )
        local_energy_batched, _ = compute_local_energy(
            helpers.rng(1),
            hamil,
            ansatz.apply,
            stacked_params,
            phys_conf,
            batch_size=2,
        )
        assert jnp.allclose(local_energy_full, local_energy_batched)


class TestComputeMeanEnergy:
    def test_mean_matches_manual_weighted_average(self):
        local_energy = jnp.array([[[1.0, 2.0, 3.0]]])
        weight = jnp.array([[[0.5, 1.0, 1.5]]])
        mean_energy, stats = pmap(compute_mean_energy)(local_energy[None], weight[None])
        expected = jnp.mean(local_energy * weight)
        assert jnp.allclose(mean_energy[0], expected)
        assert stats == {}


class TestComputeMeanEnergyTangent:
    def test_tangent_matches_manual_formula(self):
        local_energy = jnp.array([[[1.0, 2.0, 3.0, 4.0]]])
        weight = jnp.array([[[1.0, 1.0, 1.0, 1.0]]])
        log_psi_tangent = jnp.array([[[0.1, 0.2, 0.3, 0.4]]])
        gradient_mask = jnp.array([[[True, True, False, True]]])

        per_mol_state_mean_energy = jnp.mean(
            local_energy * weight, axis=-1, keepdims=True
        )
        local_energy_tangent = (
            (local_energy - per_mol_state_mean_energy) * log_psi_tangent * weight
        )
        expected = jnp.sum(
            jnp.where(gradient_mask, local_energy_tangent, 0.0)
        ) / jnp.sum(gradient_mask)

        mean_energy_tangent = pmap(compute_mean_energy_tangent)(
            local_energy[None],
            weight[None],
            log_psi_tangent[None],
            gradient_mask[None],
        )
        assert jnp.allclose(mean_energy_tangent[0], expected)
