from typing import Protocol

import jax
import jax.numpy as jnp

from ..parallel import all_device_mean
from ..types import (
    Ansatz,
    DataDict,
    Params,
    PhysicalConfiguration,
    Psi,
    Stats,
    Weight,
)
from ..utils import masked_mean, permute_matrix, triu_flat


def compute_wave_function_values(
    ansatz: Ansatz, params: Params, phys_conf: PhysicalConfiguration
) -> tuple[Psi, Stats]:
    r"""Compute the value of all WFs at samples drawn from all WFs.

    Args:
        params (dict): PyTree of WF parameters, with leading axis over the
            different WFs. Shape: :data:`[n_wfs, ...]`.
        phys_conf (~deepqmc.types.PhysicalConfiguration): input physical
            configuration samples, with leading axis over the different WFs
            the samples were drawn from.
            Shape: :data:`[n_wfs, elec_batch_size, ...]`

    Returns:
        ~deepqmc.types.Psi: the WF values
            :math:`\Psi[i, \, j, \, :] = \Psi_i({\bf r} \sim \Psi^2_j)`,
            shape: :data:`[n_wfs, n_wfs, elec_batch_size]`.
    """
    psi = jax.vmap(  # molecule_batch
        jax.vmap(  # electronic_states (wfs)
            jax.vmap(  # electronic_states (samples)
                jax.vmap(ansatz.apply, (None, 0)),  # electron_batch
                (None, 0),
            ),
            (0, None),
        ),
        (None, 0),
    )(params, phys_conf)
    return psi, {}


def compute_single_sample_psi_ratios(psi: Psi, mean_log_psi: jax.Array) -> jax.Array:
    r"""Compute all possible ratios between the WFs for a single sample.

    The mean of each WF value is subtracted before computing exponentials to avoid
    over/underflow.

    Args:
        psi (~deepqmc.types.Psi): all WF values for a single molecule
            and electron sample, shape: :data:`[electronic_states, electronic_states]`.
            mean_log_psi (jax.Array): mean log magnitude of the WFs, [electronic_states]
        mean_log_psi (jax.Array): the mean log psi values for each state,
            shape [electronic_states]

    Returns:
        jax.Array: the WF ratios
            :math:`R[i,\,j]=\frac{\Psi_i(r\sim\Psi^2_j)}{\Psi_j(r\sim\Psi^2_j)}`,
            shape :data:`[electronic_states, electronic_states]`.
    """
    shifted_log_psi = psi.log - mean_log_psi[:, None]
    # log_ratio[i, j] = log(|Psi_i(r~j)|) - log(|Psi_j(r~j)|)
    log_ratio = shifted_log_psi - jnp.diagonal(shifted_log_psi)[None, :]
    sign_ratio = psi.sign * jnp.diagonal(psi.sign)[None, :]
    return sign_ratio * jnp.exp(log_ratio)


def compute_psi_ratio(
    ansatz: Ansatz, params: Params, phys_conf: PhysicalConfiguration
) -> tuple[jax.Array, Stats]:
    r"""Compute the ratio of all wave function for a batch of samples.

    Args:
        ansatz (~deepqmc.types.Ansatz): the ansatz object.
        params (~deepqmc.types.Params): the current parameters of the Ansatz.
        phys_conf (~deepqmc.types.PhysicalConfiguration): the input to the Ansatz,
            shape: ``[mol_batch_size, electronic_states, electron_batch_size, ...]``.

    Returns:
        tuple[jax.Array, ~deepqmc.types.Stats]: the tuple of the WF ratios and overlap
            statistics.
    """
    psi, stats = compute_wave_function_values(ansatz, params, phys_conf)
    mean_log_psi = jnp.mean(
        psi.log, axis=(-1, -2)
    )  # mean over all samples from all states
    psi_ratio = jax.vmap(  # molecule_batch
        jax.vmap(compute_single_sample_psi_ratios, (-1, None), -1)  # electron_batch
    )(psi, mean_log_psi)
    return psi_ratio, stats


def symmetrize_overlap_with_clipped_geometric_mean(x: jax.Array) -> jax.Array:
    r"""Symmetrize the overlap using the clipped geometric mean of it and its transpose.

    Useful for computing an estimate of the overlap matrix using Monte Carlo samples
    from all WFs. Given input :math:`x_{ij}` this function computes:

    .. math::
        y_{ij}=\text{sign}(x_{ij}) \sqrt{\text{clamp}(0, \, x_{ij} \cdot x_{ji}, \, 1)}

    Note that if the signs of :math:`x_{ij}` and :math:`x_{ji}` differ, the clamping
    makes sure that :math:`y_{ij}` will be zero. Otherwise the two signs will agree,
    and we can use either one of them to compute the sign of :math:`y_{ij}`.

    Args:
        x: the non-symmetric overlap values:
            :math:`x_{ij}=\frac1N\sum_{{\bf r}_j\sim\Psi^2_j}\frac{\Psi_i({\bf r}_j)}
            {\Psi_j({\bf r}_j)}`
    """
    return jnp.sign(x) * jnp.sqrt(jnp.clip(x * jnp.transpose(x), 0.0))


def compute_mean_overlap(
    psi_ratio: jax.Array, weight: jax.Array
) -> tuple[jax.Array, Stats]:
    r"""Compute an estimate of the overlap matrix from WF ratios.

    Args:
        psi_ratio (jax.Array): the WF ratios
            :math:`\text{ratio}[i,\,j,\,:]=\frac{\Psi_i(r\sim\Psi^2_j)}{\Psi_j(r\sim\Psi^2_j)}`,
            shape: :data:`[n_wfs, n_wfs, elec_batch_size]`.
        weight (~deepqmc.types.Weight): the sample weights, shape
            :data:`[n_wfs, elec_batch_size]`.

    Returns:
        tuple[jax.Array, Stats]:
            tuple of the symmetric overlap matrix estimate, shaped
            ``[mol_batch_size, n_wfs, n_wfs]``, and overlap statistics.
    """
    non_symm_overlap_estimate = all_device_mean(
        weight[:, None, :, :] * psi_ratio, axis=-1
    )
    symm_overlap_estimate = jax.vmap(symmetrize_overlap_with_clipped_geometric_mean)(
        non_symm_overlap_estimate
    )
    overlap_loss = jax.vmap(triu_flat)(symm_overlap_estimate**2).sum(axis=-1).mean()
    stats = {'overlap/pairwise/mean': symm_overlap_estimate}
    return overlap_loss, stats


class OverlapGradientScaleFactory(Protocol):
    r"""Callable that computes the scaling factor of the overlap gradient."""

    def __call__(self, data: DataDict) -> jax.Array: ...


def no_scaling(data: dict) -> jax.Array:
    r"""Return unit scaling factor."""
    return jnp.array(1.0)


def scale_by_energy_gap(data: dict, min_gap_scale_factor: float = 0.1) -> jax.Array:
    r"""Scale the overlap gradient by the energy gap between the two states."""
    energy_ewm = data['energy_ewm']
    return jnp.clip(
        jnp.nan_to_num(jnp.abs(energy_ewm[:, :, None] - energy_ewm[:, None]), nan=1.0),
        min_gap_scale_factor,
        5.0,
    )


def scale_by_energy_std(data: dict, min_gap_scale_factor: float = 0.01) -> jax.Array:
    r"""Scale the overlap gradient by the std. dev. of the states' energies."""
    return jnp.clip(
        jnp.nan_to_num(data['std_ewm'].mean(axis=0), nan=5.0), min_gap_scale_factor, 5.0
    )[:, None]


def scale_by_max_gap_std(data: dict, min_gap_scale_factor: float = 0.1) -> jax.Array:
    """Scale the overlap gradient by the max of the energy gap and std. dev."""
    gap_factor = scale_by_energy_gap(data, min_gap_scale_factor)
    std_factor = scale_by_energy_std(data, min_gap_scale_factor)
    return jnp.maximum(gap_factor, std_factor)


def compute_mean_overlap_tangent(
    psi_ratio: jax.Array,
    weight: Weight,
    log_psi_tangent: jax.Array,
    ratio_gradient_mask: jax.Array,
    overlap: jax.Array,
    scale_factory: OverlapGradientScaleFactory,
    data: DataDict,
) -> jax.Array:
    r"""Compute the tangent of the overlap matrix with respect to the Ansatz parameters.

    Args:
        psi_ratio (jax.Array): the ratio of WF values.
        weight (~deepqmc.types.Weight): the weight of each sample.
        log_psi_tangent (jax.Array): the jvp of the WF values with respect to the
            parameters of the Ansatz.
        ratio_gradient_mask (jax.Array): a samplewise boolean mask to apply to the
            gradients.
        overlap (jax.Array): the overlap matrix estimate.
        scale_factory (OverlapGradientScaleFactory): function that computes the scaling
            factor of the overlap gradient.
        data (DataDict): input data passed to the ``scale_factory`` function.

    Returns:
        jax.Array: the jvp of the sum of the upper triangle of the overlap matrix with
            respect to the Ansatz parameters.
    """
    weight = weight[:, None, :, :]
    log_psi_tangent = log_psi_tangent[:, None, :, :]
    mean_psi_ratio = all_device_mean(weight * psi_ratio, axis=-1)

    # we compute the tangent of the overlap matrix only with respect to one of the
    # WF's parameters, hence we don't add the i<=>j terms from Eq. 16 of
    # Entwistle et al. Nat. Comm. 2022.
    # TODO: if we want to get rid of the ordering of electronic states,
    # we should add the permuted term
    overlap_tangent = (psi_ratio - mean_psi_ratio[..., None]) * weight * log_psi_tangent
    overlap_tangent = masked_mean(overlap_tangent, ratio_gradient_mask, axis=-1)
    overlap_tangent = 2 * overlap_tangent * mean_psi_ratio.swapaxes(-1, -2)
    overlap_tangent *= scale_factory(data)
    overlap_tangent = jax.vmap(permute_matrix)(overlap_tangent, data['ordering'])
    return jax.vmap(triu_flat)(overlap_tangent).sum(axis=-1).mean()
