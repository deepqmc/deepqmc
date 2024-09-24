import jax
import jax.numpy as jnp


def to_absolute_axis_idx(axis: int, ndim: int) -> int:
    r"""Converts a relative axis index to an absolute axis index."""
    return axis % ndim


def sampling_error(
    samples: jax.Array, walker_axis: int = -1, iteration_axis: int = 0
) -> jax.Array:
    r"""Estimate the sampling error from a set of samples using the blocking method."""
    walker_axis = to_absolute_axis_idx(walker_axis, samples.ndim)
    iteration_axis = to_absolute_axis_idx(iteration_axis, samples.ndim)
    assert walker_axis != iteration_axis
    if iteration_axis < walker_axis:
        walker_axis -= 1
    block_mean = samples.mean(axis=iteration_axis)
    return block_mean.std(axis=walker_axis) / jnp.sqrt(block_mean.shape[walker_axis])
