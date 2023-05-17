import logging

import jax.numpy as jnp
import kfac_jax
from jax import vmap

__all__ = ['make_graph_patterns']

log = logging.getLogger(__name__)


class DenseBlock(kfac_jax.DenseTwoKroneckerFactored):
    r"""
    Modification of the kfac_jax dense block.

    Expand the input to include batch dimension, if necessary.
    """

    def _update_curvature_matrix_estimate(
        self,
        state,
        estimation_data,
        ema_old,
        ema_new,
        batch_size,
    ):
        (x,) = estimation_data['inputs']
        (dy,) = estimation_data['outputs_tangent']
        if not kfac_jax.utils.first_dim_is_size(batch_size, x, dy):
            log.debug("Input of dense block doesn't have first dim of batch_size")
            log.debug(f"It's shape is {x.shape}, expanding to {(batch_size, *x.shape)}")
            x, dy = (
                jnp.tile(a[None], (batch_size, *(1 for _ in a.shape))).reshape(
                    (-1, a.shape[-1])
                )
                for a in (x, dy)
            )
            batch_size = x.size // x.shape[-1]

        return super()._update_curvature_matrix_estimate(
            state, estimation_data, ema_old, ema_new, batch_size
        )


class RepeatedDenseBlock(kfac_jax.DenseTwoKroneckerFactored):
    """Dense block that is repeatedly applied to multiple inputs (e.g. vmap)."""

    def fixed_scale(self):
        (x_shape,) = self.inputs_shapes
        return float(kfac_jax.utils.product(x_shape) // (x_shape[0] * x_shape[-1]))

    def _update_curvature_matrix_estimate(
        self,
        state,
        estimation_data,
        ema_old,
        ema_new,
        batch_size,
    ):
        estimation_data = dict(**estimation_data)
        (x,) = estimation_data['inputs']
        (dy,) = estimation_data['outputs_tangent']
        assert kfac_jax.utils.first_dim_is_size(batch_size, x, dy)

        estimation_data['inputs'] = (x.reshape([-1, x.shape[-1]]),)
        estimation_data['outputs_tangent'] = (dy.reshape([-1, dy.shape[-1]]),)
        batch_size = x.size // x.shape[-1]
        return super()._update_curvature_matrix_estimate(
            state, estimation_data, ema_old, ema_new, batch_size
        )


def _dense(x, params):
    """Compute a dense layer."""
    w, *opt_b = params
    y = jnp.matmul(x, w)
    return y if not opt_b else y + opt_b[0]


def _dense_parameter_extractor(eqns):
    """Extract all parameters from the dot_general operator."""
    for eqn in eqns:
        if eqn.primitive.name == 'dot_general':
            return dict(**eqn.params)
    raise AssertionError()


def make_dense_pattern(
    with_bias,
    in_dim=13,
    out_dim=7,
    extra_dims=None,
):
    r"""Create :class:`GraphPattern`s matching dense layers."""
    n_extra_dims = len(extra_dims or ())
    x_shape = (2, *(extra_dims or ()), in_dim)
    p_shapes = [[in_dim, out_dim], [out_dim]] if with_bias else [[in_dim, out_dim]]
    compute_func = vmap(_dense, in_axes=(0, None))
    return kfac_jax.tag_graph_matcher.GraphPattern(
        name=(
            f'repeated{n_extra_dims}_dense_with_bias'
            if with_bias
            else f'repeated{n_extra_dims}_dense_no_bias'
        ),
        tag_primitive=kfac_jax.LayerTag(f'repeated{n_extra_dims}_dense_tag', 1, 1),
        #  precedence=0 if with_bias else 1,
        compute_func=compute_func,
        parameters_extractor_func=_dense_parameter_extractor,
        example_args=[jnp.zeros(x_shape), [jnp.zeros(s) for s in p_shapes]],
    )


def make_graph_patterns():
    r"""Create deepqmc graph patterns for the KFAC optimizer."""
    custom_patterns = []
    for n_extra_dims in range(2, 0, -1):
        for with_bias in (True, False):
            custom_patterns.append(
                make_dense_pattern(
                    with_bias, extra_dims=tuple(8 + i for i in range(n_extra_dims))
                )
            )
            kfac_jax.set_default_tag_to_block_ctor(
                f'repeated{n_extra_dims}_dense_tag', RepeatedDenseBlock
            )

    kfac_jax.set_default_tag_to_block_ctor('dense_tag', DenseBlock)

    graph_patterns = (
        *(custom_patterns),
        *kfac_jax.tag_graph_matcher.DEFAULT_GRAPH_PATTERNS,
    )
    return graph_patterns
