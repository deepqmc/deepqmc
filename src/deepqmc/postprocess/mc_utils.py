import jax
import jax.numpy as jnp


def to_absolute_axis_idx(axis: int, ndim: int) -> int:
    r"""Converts a relative axis index to an absolute axis index."""
    return axis % ndim


def sampling_error(
    samples: jax.Array, walker_axis: int = -1, iteration_axis: int = 0
) -> jax.Array:
    r"""Estimate the statistical sampling error from a set of samples via blocking.

    Splits the samples into blocks along ``iteration_axis`` (each block being one
    sampling iteration), averages within each block over ``walker_axis``, and
    returns the standard error of the mean of the resulting block averages. This
    blocking approach reduces the impact of autocorrelation between consecutive
    Monte Carlo steps compared to a naive standard-error estimate over all samples.

    Args:
        samples (~jax.Array): the array of samples, containing at least a walker and
            an iteration axis.
        walker_axis (int): optional, the axis indexing independent walkers, e.g. the
            electron batch axis.
        iteration_axis (int): optional, the axis indexing the sampling iterations,
            e.g. the training or evaluation steps.

    Returns:
        ~jax.Array: the estimated sampling error, with the ``walker_axis`` and
        ``iteration_axis`` reduced out.
    """
    walker_axis = to_absolute_axis_idx(walker_axis, samples.ndim)
    iteration_axis = to_absolute_axis_idx(iteration_axis, samples.ndim)
    assert walker_axis != iteration_axis
    if iteration_axis < walker_axis:
        walker_axis -= 1
    block_mean = samples.mean(axis=iteration_axis)
    return block_mean.std(axis=walker_axis) / jnp.sqrt(block_mean.shape[walker_axis])


def clipped_mean_and_sampling_error(
    samples: jax.Array,
    lower: float = -jnp.inf,
    upper: float = jnp.inf,
    iteration_axis: int = 0,
    walker_axis: int = 1,
):
    r"""Compute the mean and sampling error of samples clipped to a given range.

    Discards samples (and ``nan`` values) outside of ``[lower, upper]`` before
    computing the mean over both the walker and iteration axes, and estimates the
    sampling error from the standard deviation of the per-walker means. Useful for
    robustly estimating observables, e.g. local energies, in the presence of rare
    outlier samples.

    Args:
        samples (~jax.Array): the array of samples, containing a walker and an
            iteration axis.
        lower (float): the lower bound of the range samples are clipped to.
        upper (float): the upper bound of the range samples are clipped to.
        iteration_axis (int): optional, the axis indexing the sampling iterations.
        walker_axis (int): optional, the axis indexing independent walkers.

    Returns:
        tuple[~jax.Array, ~jax.Array, dict]: a tuple of the clipped mean, the
        sampling error of the mean, and a dictionary with the ratio of samples that
        were kept, i.e. not clipped, under the key ``'kept_sample_ratio'``.
    """
    samples_mask = (samples < upper) & (samples > lower) & ~jnp.isnan(samples)
    all_walker_mean = (samples * samples_mask).sum(
        (iteration_axis, walker_axis)
    ) / samples_mask.sum((iteration_axis, walker_axis))
    per_walker_mean = (samples * samples_mask).sum(iteration_axis) / samples_mask.sum(
        iteration_axis
    )
    # we assume that there is at least one non-masked out sample for each walker
    sampling_error = jnp.std(
        per_walker_mean,
        (
            walker_axis - 1
            if iteration_axis < walker_axis and iteration_axis >= 0 and walker_axis > 0
            else walker_axis
        ),
    ) / jnp.sqrt(samples.shape[walker_axis])
    return all_walker_mean, sampling_error, {'kept_sample_ratio': samples_mask.mean()}


def clipped_batch_mean_and_std(
    samples: jax.Array, lower: float, upper: float, walker_axis: int = 1
) -> tuple[jax.Array, jax.Array]:
    r"""Compute the batchwise mean and std. dev. of samples clipped to a given range.

    Discards samples (and ``nan`` values) outside of ``[lower, upper]`` before
    computing the mean and standard deviation along ``walker_axis``, independently
    for each remaining batch entry, e.g. each iteration.

    Args:
        samples (~jax.Array): the array of samples.
        lower (float): the lower bound of the range samples are clipped to.
        upper (float): the upper bound of the range samples are clipped to.
        walker_axis (int): optional, the axis indexing independent walkers.

    Returns:
        tuple[~jax.Array, ~jax.Array]: a tuple of the clipped mean and standard
        deviation, with ``walker_axis`` reduced out.
    """
    samples_mask = (samples < upper) & (samples > lower) & ~jnp.isnan(samples)
    masked_samples = samples * samples_mask
    means = (masked_samples).sum(walker_axis, keepdims=True) / samples_mask.sum(
        walker_axis, keepdims=True
    )
    masked_means = jnp.where(samples_mask, means, 0)
    stds = jnp.sqrt(
        jnp.sum((masked_samples - masked_means) ** 2, axis=walker_axis)
        / samples_mask.sum(walker_axis)
    )
    return means.squeeze(walker_axis), stds
