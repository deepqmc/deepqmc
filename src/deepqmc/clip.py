import jax.numpy as jnp

from .parallel import all_device_mean, all_device_median, all_device_quantile
from .utils import log_squeeze

__all__ = ()


def median_clip_and_mask(x, clip_width, median_center, exclude_width=jnp.inf):
    clip_center = all_device_median(x) if median_center else all_device_mean(x)
    abs_diff = jnp.abs(x - clip_center)
    mad = all_device_mean(abs_diff)
    x_clip = jnp.clip(x, clip_center - clip_width * mad, clip_center + clip_width * mad)
    gradient_mask = abs_diff < exclude_width
    return x_clip, gradient_mask


def median_log_squeeze_and_mask(
    x, clip_width=1.0, quantile=0.95, exclude_width=jnp.inf
):
    x_median = all_device_median(x)
    x_diff = x - x_median
    x_abs_diff = jnp.abs(x_diff)
    quantile = all_device_quantile(x_abs_diff, quantile)
    width = clip_width * quantile
    x_clip = x_median + 2 * width * log_squeeze(x_diff / (2 * width))
    gradient_mask = x_abs_diff / quantile < exclude_width
    return x_clip, gradient_mask
