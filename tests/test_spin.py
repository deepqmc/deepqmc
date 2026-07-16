import jax.numpy as jnp

from deepqmc.loss.spin import (
    compute_mean_spin,
    compute_mean_spin_raising_tangent,
    compute_mean_spin_tangent,
    compute_spin_contributions,
    compute_spin_raising_contributions,
)
from deepqmc.parallel import pmap
from deepqmc.utils import tree_stack


class TestComputeSpinContributions:
    def test_shape_and_finite(self, helpers):
        hamil = helpers.hamil()
        ansatz, params = helpers.create_ansatz(hamil)
        phys_conf, _ = helpers.batch_phys_conf(
            hamil, mol_batch=2, n_states=2, elec_batch=3
        )
        stacked_params = tree_stack([params, params])
        spin_contrib = compute_spin_contributions(
            hamil, ansatz, stacked_params, phys_conf
        )
        assert spin_contrib.shape == (2, 2, 3)
        assert jnp.all(jnp.isfinite(spin_contrib))

    def test_states_subset_matches_full(self, helpers):
        hamil = helpers.hamil()
        ansatz, params = helpers.create_ansatz(hamil)
        phys_conf, _ = helpers.batch_phys_conf(
            hamil, mol_batch=2, n_states=2, elec_batch=3
        )
        stacked_params = tree_stack([params, params])
        spin_contrib_full = compute_spin_contributions(
            hamil, ansatz, stacked_params, phys_conf
        )
        spin_contrib_state0 = compute_spin_contributions(
            hamil, ansatz, stacked_params, phys_conf, states=[0]
        )
        assert spin_contrib_state0.shape == (2, 1, 3)
        assert jnp.allclose(spin_contrib_state0[:, 0], spin_contrib_full[:, 0])

    def test_single_state_batch_defaults_to_state_zero(self, helpers):
        hamil = helpers.hamil()
        ansatz, params = helpers.create_ansatz(hamil)
        phys_conf, _ = helpers.batch_phys_conf(
            hamil, mol_batch=2, n_states=1, elec_batch=3
        )
        stacked_params = tree_stack([params])
        spin_contrib = compute_spin_contributions(
            hamil, ansatz, stacked_params, phys_conf
        )
        spin_contrib_explicit = compute_spin_contributions(
            hamil, ansatz, stacked_params, phys_conf, states=[0]
        )
        assert spin_contrib.shape == (2, 1, 3)
        assert jnp.allclose(spin_contrib, spin_contrib_explicit)


class TestComputeMeanSpin:
    def test_mean_and_stats(self):
        spin_contributions = jnp.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        weight = jnp.array([[[1.0, 2.0, 1.0], [0.5, 1.0, 1.5]]])

        mean_spin, stats = pmap(lambda sc, w: compute_mean_spin(sc, w))(
            spin_contributions[None], weight[None]
        )

        assert jnp.allclose(mean_spin[0], jnp.mean(spin_contributions * weight))

        def weighted_mean_manual(x, w):
            return jnp.sum(x * w) / jnp.sum(w)

        def weighted_std_manual(x, w):
            mean = weighted_mean_manual(x, w)
            var = jnp.sum(w * (x - mean) ** 2) / jnp.sum(w)
            return jnp.sqrt(var)

        expected_mean = jnp.array(
            [
                weighted_mean_manual(spin_contributions[0, 0], weight[0, 0]),
                weighted_mean_manual(spin_contributions[0, 1], weight[0, 1]),
            ]
        )
        expected_std = jnp.array(
            [
                weighted_std_manual(spin_contributions[0, 0], weight[0, 0]),
                weighted_std_manual(spin_contributions[0, 1], weight[0, 1]),
            ]
        )
        assert jnp.allclose(stats['spin/mean'][0, 0], expected_mean)
        assert jnp.allclose(stats['spin/std'][0, 0], expected_std)

    def test_states_subset_selects_correct_weight_column(self):
        spin_contributions = jnp.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        weight = jnp.array([[[1.0, 2.0, 1.0], [0.5, 1.0, 1.5]]])
        sc_state1 = spin_contributions[:, 1:2, :]

        mean_spin, stats = pmap(lambda sc, w: compute_mean_spin(sc, w, states=[1]))(
            sc_state1[None], weight[None]
        )

        expected_mean = jnp.sum(spin_contributions[0, 1] * weight[0, 1]) / jnp.sum(
            weight[0, 1]
        )
        assert jnp.allclose(mean_spin[0], jnp.mean(sc_state1 * weight[:, 1:2, :]))
        assert jnp.allclose(stats['spin/mean'][0, 0, 0], expected_mean)


class TestComputeMeanSpinTangent:
    def test_tangent_matches_manual_formula(self):
        spin_contributions = jnp.array([[[1.0, 3.0], [2.0, 4.0]]])
        weight = jnp.array([[[1.0, 1.0], [2.0, 1.0]]])
        log_psi_tangent = jnp.array([[[0.5, -0.5], [1.0, 0.0]]])
        gradient_mask = jnp.array([[[True, True], [True, False]]])

        tangent = pmap(
            lambda sc, w, lpt, gm: compute_mean_spin_tangent(sc, w, lpt, gm)
        )(
            spin_contributions[None],
            weight[None],
            log_psi_tangent[None],
            gradient_mask[None],
        )

        per_mol_state_mean = jnp.mean(
            spin_contributions * weight, axis=-1, keepdims=True
        )
        contributions_tangent = (
            (spin_contributions - per_mol_state_mean) * log_psi_tangent * weight
        )
        expected = jnp.sum(
            jnp.where(gradient_mask, contributions_tangent, 0.0)
        ) / jnp.sum(gradient_mask)

        assert jnp.allclose(tangent[0], expected)


class TestComputeSpinRaisingContributions:
    def test_shape_and_finite(self, helpers):
        hamil = helpers.hamil()
        ansatz, params = helpers.create_ansatz(hamil)
        phys_conf, _ = helpers.batch_phys_conf(
            hamil, mol_batch=2, n_states=2, elec_batch=3
        )
        stacked_params = tree_stack([params, params])
        rng = helpers.rng(2)
        result = compute_spin_raising_contributions(
            rng, hamil, ansatz, phys_conf, stacked_params
        )
        assert result.shape == (2, 2, 3)
        assert jnp.all(jnp.isfinite(result))

    def test_determinism_and_rng_dependence(self, helpers):
        hamil = helpers.hamil()
        ansatz, params = helpers.create_ansatz(hamil)
        phys_conf, _ = helpers.batch_phys_conf(
            hamil, mol_batch=2, n_states=2, elec_batch=3
        )
        stacked_params = tree_stack([params, params])

        rng0 = helpers.rng(0)
        result0a = compute_spin_raising_contributions(
            rng0, hamil, ansatz, phys_conf, stacked_params
        )
        result0b = compute_spin_raising_contributions(
            rng0, hamil, ansatz, phys_conf, stacked_params
        )
        assert jnp.allclose(result0a, result0b)

        rng1 = helpers.rng(1)
        result1 = compute_spin_raising_contributions(
            rng1, hamil, ansatz, phys_conf, stacked_params
        )
        assert not jnp.allclose(result0a, result1)

    def test_batch_size_matches_default(self, helpers):
        hamil = helpers.hamil()
        ansatz, params = helpers.create_ansatz(hamil)
        phys_conf, _ = helpers.batch_phys_conf(
            hamil, mol_batch=2, n_states=2, elec_batch=3
        )
        stacked_params = tree_stack([params, params])
        rng = helpers.rng(2)
        result_default = compute_spin_raising_contributions(
            rng, hamil, ansatz, phys_conf, stacked_params
        )
        result_batched = compute_spin_raising_contributions(
            rng, hamil, ansatz, phys_conf, stacked_params, batch_size=3
        )
        assert jnp.allclose(result_default, result_batched)


class TestComputeMeanSpinRaisingTangent:
    def test_tangent_matches_manual_formula(self):
        spin_raising_contributions = jnp.array([[[2.0, 4.0], [1.0, 3.0]]])
        spin_raising_tangent = jnp.array([[[0.1, 0.2], [0.3, 0.4]]])
        weight = jnp.array([[[1.0, 2.0], [1.0, 1.0]]])
        log_psi_tangent = jnp.array([[[0.5, -0.5], [1.0, 0.0]]])
        gradient_mask = jnp.array([[[True, True], [True, False]]])

        tangent = pmap(
            lambda src, srt, w, lpt, gm: compute_mean_spin_raising_tangent(
                src, srt, w, lpt, gm
            )
        )(
            spin_raising_contributions[None],
            spin_raising_tangent[None],
            weight[None],
            log_psi_tangent[None],
            gradient_mask[None],
        )

        per_mol_state_mean = jnp.mean(
            spin_raising_contributions * weight, axis=-1, keepdims=True
        )
        self_adjoint_tangent = (
            spin_raising_contributions - per_mol_state_mean
        ) * log_psi_tangent
        total_tangent = (
            per_mol_state_mean
            * weight
            * (2 * self_adjoint_tangent + spin_raising_tangent)
        )
        expected = jnp.sum(jnp.where(gradient_mask, total_tangent, 0.0)) / jnp.sum(
            gradient_mask
        )

        assert jnp.allclose(tangent[0], expected)
