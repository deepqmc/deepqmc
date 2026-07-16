import jax
import jax.numpy as jnp

from deepqmc.loss.overlap import (
    compute_mean_overlap,
    compute_mean_overlap_tangent,
    compute_psi_ratio,
    compute_wave_function_values,
    no_scaling,
    scale_by_energy_gap,
    scale_by_energy_std,
    scale_by_max_gap_std,
    symmetrize_overlap_with_clipped_geometric_mean,
)
from deepqmc.parallel import pmap
from deepqmc.utils import tree_stack


def diag_log_psi(ansatz, stacked_params, phys_conf):
    def single(params, one_phys_conf):
        return ansatz.apply(params, one_phys_conf).log

    return jax.vmap(jax.vmap(jax.vmap(single, (None, 0)), (0, 0)), (None, 0))(
        stacked_params, phys_conf
    )


class TestComputePsiRatio:
    def test_shapes_and_diagonal(self, helpers):
        hamil = helpers.hamil()
        ansatz, params = helpers.create_ansatz(hamil)
        phys_conf, weight = helpers.batch_phys_conf(
            hamil, mol_batch=2, n_states=2, elec_batch=3
        )
        stacked_params = tree_stack([params, params])

        psi, _ = compute_wave_function_values(ansatz, stacked_params, phys_conf)
        assert psi.sign.shape == psi.log.shape == (2, 2, 2, 3)

        psi_ratio, _ = compute_psi_ratio(ansatz, stacked_params, phys_conf)
        assert psi_ratio.shape == (2, 2, 2, 3)
        assert psi_ratio.dtype == jnp.float64
        assert jnp.isfinite(psi_ratio).all()
        diag = jnp.diagonal(psi_ratio, axis1=1, axis2=2)
        assert jnp.allclose(diag, 1.0)


class TestSymmetrizeOverlap:
    def test_same_sign_below_one(self):
        x = jnp.array([[1.0, 0.3], [0.2, 1.0]])
        y = symmetrize_overlap_with_clipped_geometric_mean(x)
        expected = jnp.array([[1.0, jnp.sqrt(0.06)], [jnp.sqrt(0.06), 1.0]])
        assert jnp.allclose(y, expected)

    def test_opposite_sign_clips_to_zero(self):
        x = jnp.array([[1.0, -0.4], [0.3, 1.0]])
        y = symmetrize_overlap_with_clipped_geometric_mean(x)
        assert jnp.allclose(y[0, 1], 0.0)
        assert jnp.allclose(y[1, 0], 0.0)
        assert jnp.allclose(y[0, 0], 1.0)
        assert jnp.allclose(y[1, 1], 1.0)

    def test_product_above_one_is_not_clipped(self):
        x = jnp.array([[1.0, 2.0], [3.0, 1.0]])
        y = symmetrize_overlap_with_clipped_geometric_mean(x)
        expected = jnp.array([[1.0, jnp.sqrt(6.0)], [jnp.sqrt(6.0), 1.0]])
        assert jnp.allclose(y, expected)

    def test_mixed_3x3(self):
        x = jnp.array(
            [
                [1.0, 0.3, -0.5],
                [0.2, 1.0, 0.4],
                [0.6, 0.5, 1.0],
            ]
        )
        y = symmetrize_overlap_with_clipped_geometric_mean(x)
        expected = jnp.array(
            [
                [1.0, jnp.sqrt(0.06), 0.0],
                [jnp.sqrt(0.06), 1.0, jnp.sqrt(0.2)],
                [0.0, jnp.sqrt(0.2), 1.0],
            ]
        )
        assert jnp.allclose(y, expected)


class TestComputeMeanOverlap:
    def test_weighted_mean_and_symmetrization(self):
        psi_ratio = jnp.array(
            [
                [
                    [[1.0, 1.0], [0.2, 0.4]],
                    [[0.3, 0.5], [1.0, 1.0]],
                ]
            ]
        )
        weight = jnp.array([[[1.0, 1.0], [0.8, 1.2]]])
        overlap_loss, stats = pmap(compute_mean_overlap)(psi_ratio[None], weight[None])
        expected_symm = jnp.array([[1.0, jnp.sqrt(0.128)], [jnp.sqrt(0.128), 1.0]])
        assert jnp.allclose(overlap_loss[0], 0.128)
        assert stats['overlap/pairwise/mean'][0].shape == (1, 2, 2)
        assert jnp.allclose(stats['overlap/pairwise/mean'][0, 0], expected_symm)


class TestOverlapScaleFactories:
    def test_no_scaling_ignores_data(self):
        assert no_scaling({'anything': 1.0}) == 1.0
        assert no_scaling({}) == 1.0

    def test_scale_by_energy_gap(self):
        energy_ewm = jnp.array([[0.0, 0.05, 10.0]])
        gap = scale_by_energy_gap({'energy_ewm': energy_ewm})
        expected = jnp.array([[[0.1, 0.1, 5.0], [0.1, 0.1, 5.0], [5.0, 5.0, 0.1]]])
        assert jnp.allclose(gap, expected)

    def test_scale_by_energy_gap_nan_energy_falls_back_to_one(self):
        energy_ewm = jnp.array([[0.0, jnp.nan]])
        gap = scale_by_energy_gap({'energy_ewm': energy_ewm})
        expected = jnp.array([[[0.1, 1.0], [1.0, 1.0]]])
        assert jnp.allclose(gap, expected)

    def test_scale_by_energy_std(self):
        std_ewm = jnp.array([[0.02, 0.2, 0.005], [0.04, 0.3, 0.6]])
        std = scale_by_energy_std({'std_ewm': std_ewm})
        expected = jnp.array([[0.03], [0.25], [0.3025]])
        assert jnp.allclose(std, expected)

    def test_scale_by_energy_std_nan_falls_back_to_five(self):
        std_ewm = jnp.array([[0.02, jnp.nan], [0.04, 0.5]])
        std = scale_by_energy_std({'std_ewm': std_ewm})
        expected = jnp.array([[0.03], [5.0]])
        assert jnp.allclose(std, expected)

    def test_scale_by_max_gap_std_is_elementwise_max(self):
        data = {
            'energy_ewm': jnp.array([[0.0, 0.05, 10.0]]),
            'std_ewm': jnp.array([[0.02, 0.2, 0.005], [0.04, 0.3, 0.6]]),
        }
        gap = scale_by_energy_gap(data, 0.1)
        std = scale_by_energy_std(data, 0.1)
        expected = jnp.maximum(gap, std)
        actual = scale_by_max_gap_std(data, 0.1)
        assert jnp.allclose(actual, expected)


class TestComputeMeanOverlapTangent:
    def test_tangent_vanishes_iff_wave_function_is_constant(self, helpers):
        hamil = helpers.hamil()
        ansatz, params = helpers.create_ansatz(hamil)
        phys_conf, weight = helpers.batch_phys_conf(
            hamil, mol_batch=2, n_states=2, elec_batch=3
        )
        params2 = jax.tree.map(lambda x: x * 1.05 + 0.01, params)
        stacked_params = tree_stack([params, params2])

        psi_ratio, _ = compute_psi_ratio(ansatz, stacked_params, phys_conf)
        _, stats = pmap(compute_mean_overlap)(psi_ratio[None], weight[None])
        overlap = stats['overlap/pairwise/mean'][0]
        ratio_gradient_mask = jnp.ones_like(psi_ratio, dtype=bool)
        ordering = jnp.broadcast_to(jnp.arange(2), (2, 2))

        def tangent_call(psi_ratio, weight, log_psi_tangent, mask, overlap, ordering):
            return compute_mean_overlap_tangent(
                psi_ratio,
                weight,
                log_psi_tangent,
                mask,
                overlap,
                no_scaling,
                {'ordering': ordering},
            )

        _, log_psi_tangent = jax.jvp(
            lambda p: diag_log_psi(ansatz, p, phys_conf),
            (stacked_params,),
            (stacked_params,),
        )
        result = pmap(tangent_call)(
            psi_ratio[None],
            weight[None],
            log_psi_tangent[None],
            ratio_gradient_mask[None],
            overlap[None],
            ordering[None],
        )
        assert jnp.isfinite(result).all()
        assert jnp.abs(result[0]) > 0.0

        zero_tangent = jax.tree.map(jnp.zeros_like, stacked_params)
        _, log_psi_tangent_zero = jax.jvp(
            lambda p: diag_log_psi(ansatz, p, phys_conf),
            (stacked_params,),
            (zero_tangent,),
        )
        result_zero = pmap(tangent_call)(
            psi_ratio[None],
            weight[None],
            log_psi_tangent_zero[None],
            ratio_gradient_mask[None],
            overlap[None],
            ordering[None],
        )
        assert jnp.allclose(result_zero, 0.0)
