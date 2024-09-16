from typing import Protocol, Union

import jax
import jax.numpy as jnp

from ..parallel import all_device_mean, all_device_median, all_device_quantile
from ..types import Energy
from ..utils import log_squeeze

__all__ = ()


class LocalEnergyClipAndMaskFn(Protocol):
    r"""Protocol for functions masking a single electron batch of local energies."""

    def __call__(self, __local_energy: Energy) -> tuple[Energy, jax.Array]: ...


class PsiRatioClipAndMaskFn(Protocol):
    r"""Protocol for functions masking a single electron batch of WF ratios."""

    def __call__(self, psi_ratio: jax.Array) -> tuple[jax.Array, jax.Array]: ...


def clip_local_energy(
    clip_mask_fn: LocalEnergyClipAndMaskFn, local_energy: Energy
) -> tuple[Energy, jax.Array]:
    r"""Apply a clipping function to the local energies.

    The clipping function is twice ``vmap``ed: over the molecule batch, and
    electronic state dimensions.

    Args:
        clip_mask_fn (Callable[[jax.Array], (jax.Array, jax.Array)]): function taking
            as input an electron batch of local energies and returning a tuple of the
            clipped local energies and an identically shaped boolean mask array to be
            applied to the gradients.
        local_energy (jax.Array): the electron batch of local energies, shape:
            ``[mol_batch_size, electronic_states, electron_batch_size // device_count]
            ``.
    """
    return jax.vmap(jax.vmap(clip_mask_fn))(local_energy)


def clip_psi_ratio(
    clip_mask_fn: PsiRatioClipAndMaskFn, psi_ratio: jax.Array
) -> tuple[jax.Array, jax.Array]:
    r"""Apply a clipping function to the wave function ratios.

    The clipping function is thrice ``vmap``ed: over the molecule batch, and the two
    electronic state dimensions of the wave function ratio array:
    :math:`\text{ratio}[i,\,j,\,:]=\frac{\Psi_i(r\sim\Psi^2_j)}{\Psi_j(r\sim\Psi^2_j)}`.

    Args:
        clip_mask_fn (Callable[[jax.Array, jax.Array], (jax.Array, jax.Array)]):
            function taking as input an electron batch of the ratios between two WFs,
            the estimated overlap of these two WFs, and returning a tuple of the clipped
            ratios and an identically shaped boolean mask array to be applied to the
            gradients.
        psi_ratio (jax.Array): the electron batch of psi_ratios, shape:
            ``[mol_batch_size, electronic_states, electronic_states,
            electron_batch_size // device_count]``.
        overlap (jax.Array): the overlap estimate of the electronic states, shape:
            ``[mol_batch_size, electronic_states, electronic_states]``.
    """
    return jax.vmap(jax.vmap(jax.vmap(clip_mask_fn)))(psi_ratio)


def median_clip_and_mask(
    x: jax.Array, clip_width: float, median_center: bool, exclude_width: float = jnp.inf
) -> tuple[jax.Array, jax.Array]:
    clip_center = all_device_median(x) if median_center else all_device_mean(x)
    abs_diff = jnp.abs(x - clip_center)
    mad = all_device_mean(abs_diff)
    x_clip = jnp.clip(x, clip_center - clip_width * mad, clip_center + clip_width * mad)
    gradient_mask = abs_diff < exclude_width
    return x_clip, gradient_mask


def median_log_squeeze_and_mask(
    x: jax.Array,
    clip_width: float = 1.0,
    quantile: Union[float, jax.Array] = 0.95,
    exclude_width: float = jnp.inf,
) -> tuple[jax.Array, jax.Array]:
    x_median = all_device_median(x)
    x_diff = x - x_median
    x_abs_diff = jnp.abs(x_diff)
    quantile = all_device_quantile(x_abs_diff, quantile)
    width = clip_width * quantile
    x_clip = x_median + 2 * width * log_squeeze(x_diff / (2 * width))
    gradient_mask = x_abs_diff / quantile < exclude_width
    return x_clip, gradient_mask


def psi_ratio_clip_and_mask(
    psi_ratio: jax.Array,
    *,
    clip_width: float = 10.0,
    exclude_width: float = jnp.inf,
) -> tuple[jax.Array, jax.Array]:
    r"""Clips WF ratios of a single batch of electron position samples.

    Args:
        ratio ([electron_batch_size]): ratio of log WF values:
            :math:`\frac{\Psi_i({\bf r}_j)}{\Psi_j({\bf r}_j)}`.
        overlap (float): the approximate overlap of the two WFs:
            :math:`S_{ij}`.
        clip_width (float): clip width to use when clipping ratio
        exclude_width (float): default: :data:`jnp.inf`, deviation threshold above which
            outlier ratios are excluded from the overlap gradient computation.

    Returns:
        tuple of the clipped WF ratios and the gradient mask.
    """
    clip_center = all_device_median(psi_ratio)
    deviation = jnp.abs(psi_ratio - clip_center)
    # TODO: check if using MAD to compute sigma (like in energy clipping)
    # would also work
    sigma = all_device_median(deviation)
    clipped_ratio = jnp.clip(
        psi_ratio,
        clip_center - clip_width * sigma,
        clip_center + clip_width * sigma,
    )
    ratio_gradient_mask = jnp.abs(psi_ratio - clip_center) < exclude_width
    return clipped_ratio, ratio_gradient_mask
