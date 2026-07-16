import jax.numpy as jnp

from deepqmc.postprocess.mc_utils import (
    clipped_batch_mean_and_std,
    clipped_mean_and_sampling_error,
    sampling_error,
)


class TestSamplingError:
    def test_sampling_error(self):
        samples = jnp.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        error = sampling_error(samples, walker_axis=-1, iteration_axis=0)
        assert jnp.allclose(error, 0.4714045, atol=1e-6)


class TestClippedMeanAndSamplingError:
    def test_outlier_is_excluded(self):
        samples = jnp.array([[1.0, 2.0, 3.0], [100.0, 4.0, 5.0]])
        mean, error, stats = clipped_mean_and_sampling_error(samples, -10.0, 10.0)
        assert jnp.allclose(mean, 3.0)
        assert jnp.allclose(stats['kept_sample_ratio'], 5 / 6)
        assert jnp.allclose(error, 0.72008, atol=1e-5)


class TestClippedBatchMeanAndStd:
    def test_outlier_is_excluded(self):
        samples = jnp.array([[1.0, 2.0, 3.0], [100.0, 4.0, 5.0]])
        means, stds = clipped_batch_mean_and_std(samples, -10.0, 10.0)
        assert jnp.allclose(means, jnp.array([2.0, 4.5]))
        assert jnp.allclose(stds, jnp.array([jnp.sqrt(2 / 3), 0.5]))
