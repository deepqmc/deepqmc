import dataclasses
import logging

import jax.numpy as jnp
import kfac_jax

from .types import Batch

__all__ = ['layer_tag_to_block_ctor', 'batch_size_extractor']

log = logging.getLogger(__name__)


def _replace_inputs_and_output_tangents(estimation_data, x, dy):
    return dataclasses.replace(
        estimation_data,
        primals=dataclasses.replace(estimation_data.primals, inputs=(x,)),
        tangents=dataclasses.replace(estimation_data.tangents, outputs=(dy,)),
    )


def _dense_expand_batch(estimation_data, batch_size):
    r"""Give a plain dense layer's inputs/output-tangents a batch_size leading dim.

    Some parts of the ansatz (e.g. nucleus embeddings that only depend on nuclear
    charges) do not depend on the sampled electron batch, so under the outer
    jax.vmap over the batch (see loss_function.py) they never pick up a batch axis.
    kfac_jax's automatic layer registration assumes every dense layer's inputs and
    output-tangents already carry a leading batch_size axis, and otherwise raises
    an assertion error. We detect this case and instead tile the values across a
    new batch axis (equivalent to redundantly recomputing the same batch-invariant
    value for every sample), then flatten that axis together with any other
    leading (non-batch) axes, since :class:`kfac_jax.DenseTwoKroneckerFactored`
    only accepts rank-2 (batch, feature) inputs. The returned batch size is
    adjusted to match the flattened row count.
    """
    (x,) = estimation_data.primals.inputs
    (dy,) = estimation_data.tangents.outputs
    if kfac_jax.utils.first_dim_is_size(batch_size, x, dy):
        return estimation_data, batch_size
    log.debug(
        f"Dense layer input doesn't have a leading batch_size dimension, "
        f'got shape {x.shape}, tiling and flattening to match batch_size {batch_size}.'
    )

    def expand(a):
        tiled = jnp.tile(a[None], (batch_size, *(1 for _ in a.shape)))
        return tiled.reshape((-1, a.shape[-1]))

    x, dy = expand(x), expand(dy)
    return _replace_inputs_and_output_tangents(estimation_data, x, dy), x.shape[0]


def _repeated_dense_expand_batch(estimation_data, batch_size):
    r"""Give a repeated-dense layer's inputs/output-tangents a batch_size leading dim.

    Like :func:`_dense_expand_batch`, but for
    :class:`kfac_jax.RepeatedDenseKroneckerFactored`, which natively supports extra
    leading "repeat" axes between the batch and feature axes (e.g. our correctly
    batched (batch_size, n_nuc, feature) layers). We therefore only need to
    introduce the missing batch axis, without flattening or adjusting batch_size.
    """
    (x,) = estimation_data.primals.inputs
    (dy,) = estimation_data.tangents.outputs
    if kfac_jax.utils.first_dim_is_size(batch_size, x, dy):
        return estimation_data
    log.debug(
        f"Repeated dense layer input doesn't have a leading batch_size dimension, "
        f'got shape {x.shape}, tiling to {(batch_size, *x.shape)}.'
    )
    tile = lambda a: jnp.tile(a[None], (batch_size, *(1 for _ in a.shape)))
    return _replace_inputs_and_output_tangents(estimation_data, tile(x), tile(dy))


class DenseBlock(kfac_jax.DenseTwoKroneckerFactored):
    r"""A dense curvature block tolerant of batch-invariant layer inputs.

    See :func:`_dense_expand_batch` for why this is necessary.
    """

    @kfac_jax.utils.auto_scope_method
    def update_curvature_matrix_estimate(
        self, state, estimation_data, ema_old, ema_new, identity_weight, batch_size
    ):
        estimation_data, batch_size = _dense_expand_batch(estimation_data, batch_size)
        return super().update_curvature_matrix_estimate(
            state, estimation_data, ema_old, ema_new, identity_weight, batch_size
        )


class RepeatedDenseBlock(kfac_jax.RepeatedDenseKroneckerFactored):
    r"""A repeated-dense curvature block tolerant of batch-invariant layer inputs.

    See :func:`_repeated_dense_expand_batch` for why this is necessary.
    """

    @kfac_jax.utils.auto_scope_method
    def update_curvature_matrix_estimate(
        self, state, estimation_data, ema_old, ema_new, identity_weight, batch_size
    ):
        estimation_data = _repeated_dense_expand_batch(estimation_data, batch_size)
        return super().update_curvature_matrix_estimate(
            state, estimation_data, ema_old, ema_new, identity_weight, batch_size
        )


layer_tag_to_block_ctor = {
    'dense': DenseBlock,
    'repeated_dense': RepeatedDenseBlock,
}
r"""Overrides of kfac_jax's default curvature blocks, passed to
:class:`kfac_jax.Optimizer` as ``layer_tag_to_block_ctor``."""


def batch_size_extractor(batch: Batch) -> int:
    r"""Compute the batch size for KFAC.

    KFAC requires a single batch dimension, we therefore flatten our batches resulting
    in batch dimensions that are a product of our various (molecule, electron)
    batch sizes. Note that each parameter receives gradients only from its samples,
    therefore the electronic state dimension is not included in this product.

    Args:
        batch (~deepqmc.types.Batch): a tuple containing a physical configuration,
            a set of sample weights and auxiliary data.

    Returns:
        int: the product of the molecule and electron batch size dimensions.
    """
    _, weights, _ = batch
    # product of the molecule and electron batch dims
    return weights.shape[0] * weights.shape[2]
