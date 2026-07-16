import jax.numpy as jnp

from deepqmc.loss.clip import (
    clip_local_energy,
    clip_psi_ratio,
    median_clip_and_mask,
    median_log_squeeze_and_mask,
    psi_ratio_clip_and_mask,
)
from deepqmc.parallel import pmap


def trivial_clip_mask_fn(x):
    return jnp.clip(x, -1.0, 1.0), jnp.ones_like(x, dtype=bool)


class TestVmappedClipping:
    def test_clip_local_energy(self):
        local_energy = jnp.array([[[0.5, 2.0, -3.0]]])
        clipped, mask = clip_local_energy(trivial_clip_mask_fn, local_energy)
        assert jnp.allclose(clipped, jnp.array([[[0.5, 1.0, -1.0]]]))
        assert mask.all()

    def test_clip_psi_ratio(self):
        psi_ratio = jnp.array([[[[0.5, 2.0, -3.0]]]])
        clipped, mask = clip_psi_ratio(trivial_clip_mask_fn, psi_ratio)
        assert jnp.allclose(clipped, jnp.array([[[[0.5, 1.0, -1.0]]]]))
        assert mask.all()


class TestMedianClipAndMask:
    def test_outlier_is_clipped_and_masked(self):
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 100.0])[None]
        clipped, mask = pmap(
            lambda x: median_clip_and_mask(
                x, clip_width=1.0, median_center=True, exclude_width=5.0
            )
        )(x)
        assert jnp.allclose(clipped[0], jnp.array([1.0, 2.0, 3.0, 4.0, 23.2]))
        assert jnp.array_equal(mask[0], jnp.array([True, True, True, True, False]))


class TestMedianLogSqueezeAndMask:
    def test_inlier_is_left_almost_unchanged(self):
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 100.0])[None]
        squeezed, mask = pmap(
            lambda x: median_log_squeeze_and_mask(x, clip_width=1.0, quantile=0.95)
        )(x)
        assert jnp.allclose(squeezed[0, 2], 3.0)
        assert jnp.allclose(squeezed[0, 1], 2.0)
        assert jnp.allclose(squeezed[0, 3], 4.0)
        assert mask[0].all()

    def test_outlier_is_squeezed_not_hard_clipped(self):
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 100.0])[None]
        squeezed, mask = pmap(
            lambda x: median_log_squeeze_and_mask(x, clip_width=1.0, quantile=0.95)
        )(x)
        median = 3.0
        width = 1.0 * 78.0
        z = 97.0 / (2 * width)
        log_squeeze_z = jnp.sign(z) * jnp.log1p(
            (jnp.abs(z) + 0.5 * z**2 + jnp.abs(z) ** 3) / (1 + z**2)
        )
        expected_outlier = median + 2 * width * log_squeeze_z
        assert jnp.allclose(squeezed[0, 4], expected_outlier)
        assert squeezed[0, 4] < median + 2 * width
        assert squeezed[0, 4] > median + 1.0 * width
        assert mask[0].all()

    def test_exclude_width_masks_far_outlier(self):
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 100.0])[None]
        squeezed, mask = pmap(
            lambda x: median_log_squeeze_and_mask(
                x, clip_width=1.0, quantile=0.95, exclude_width=1.0
            )
        )(x)
        assert jnp.array_equal(mask[0], jnp.array([True, True, True, True, False]))

    def test_default_arguments_give_all_true_mask(self):
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])[None]
        squeezed, mask = pmap(lambda x: median_log_squeeze_and_mask(x))(x)
        assert squeezed[0].shape == x[0].shape
        assert mask[0].all()


class TestPsiRatioClipAndMask:
    def test_degenerate_sigma_clips_everything_to_center(self):
        x = jnp.array([1.0, 1.0, 1.0, 1.0, 10.0])[None]
        clipped, mask = pmap(
            lambda x: psi_ratio_clip_and_mask(x, clip_width=2.0, exclude_width=3.0)
        )(x)
        assert jnp.allclose(clipped[0], jnp.ones(5))
        assert jnp.array_equal(mask[0], jnp.array([True, True, True, True, False]))
