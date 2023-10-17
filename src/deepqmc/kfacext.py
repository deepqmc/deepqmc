import logging
from typing import Tuple

import jax.numpy as jnp
import kfac_jax
from jax import lax, vmap
from kfac_jax._src.utils import (
    get_special_case_zero_inv,
    psd_inv_cholesky,
    psd_matrix_norm,
    types,
)

__all__ = ['make_graph_patterns']

log = logging.getLogger(__name__)


class DenseBlock(kfac_jax.DenseTwoKroneckerFactored):
    r"""
    Modification of the kfac_jax dense block.

    Expand the input to include batch dimension, if necessary.
    """

    def fixed_scale(self):
        (x_shape,) = self.inputs_shapes
        assert len(x_shape) == 2
        return x_shape[0]

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
            estimation_data['inputs'], estimation_data['outputs_tangent'] = (x,), (dy,)

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
    x_shape = (*(extra_dims or ()), 11, in_dim)
    p_shapes = [[in_dim, out_dim], [out_dim]] if with_bias else [[in_dim, out_dim]]
    compute_func = _dense
    for _ in range(n_extra_dims):
        compute_func = vmap(compute_func, in_axes=(0, None))
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


Array = types.Array
Numeric = types.Numeric


def pi_adjusted_kronecker_inverse(
    *arrays: Array,
    damping: Numeric,
) -> Tuple[Array, ...]:
    """Computes pi-adjusted factored damping inverses.

    The inverse of `a_1 kron a_2 kron ... kron a_n + damping * I` is not Kronecker
    factored in general, because of the added identity. [1] proposed a pi-adjusted
    factored damping approach to approximate the inverse as a Kronecker product.
    [2] generalized this approach from two to tree factors, and [3] generalized it
    to arbitrary numbers of factors. This function implements the generalized
    approach.

    [1] - https://arxiv.org/abs/1503.05671
    [2] - https://openreview.net/forum?id=SkkTMpjex
    [3] - https://ui.adsabs.harvard.edu/abs/2021arXiv210602925R/abstract

    Args:
      *arrays: A list of matrices, vectors (which are interpreted
        as representing the diagonal of a matrix) or scalars (which are
        interpreted as being a 1x1 matrix). All matrices must be PSD.
      damping: The weight of the identity added to the Kronecker product.

    Returns:
      A list of factors with the same length as the input `arrays` whose Kronecker
      product approximates the inverse of `a_1 kron ... kron a_n + damping * I`.
    """
    # The implementation writes each single factor as `c_i u_i`, where the matrix
    # `u_i` is such that `trace(u_i) / dim(u_i) = 1`. We then factor out all the
    # scalar factors `c_i` into a single overall scaling coefficient and
    # distribute the damping to each single non-scalar factor `u_i` equally before
    # inverting them.

    norm_type = 'avg_trace'

    norms = [psd_matrix_norm(a, norm_type=norm_type) for a in arrays]

    # Compute the normalized factors `u_i`, such that Trace(u_i) / dim(u_i) = 1
    us = [ai / ni for ai, ni in zip(arrays, norms)]

    # kron(arrays) = c * kron(us)

    c = jnp.prod(jnp.array(norms))

    damping = damping.astype(
        c.dtype
    )  # pytype: disable=attribute-error  # numpy-scalars

    def regular_inverse() -> Tuple[Array, ...]:
        non_scalars = sum(1 if a.size != 1 else 0 for a in arrays)

        # We distribute the overall scale over each factor, including scalars
        if non_scalars == 0:
            # In the case where all factors are scalar we need to add the damping
            c_k = jnp.power(c + damping, 1.0 / len(arrays))
        else:
            # We distribute the damping only inside the non-scalar factors
            d_hat = jnp.power(damping / c, 1.0 / non_scalars)
            c_k = jnp.power(c, 1.0 / len(arrays))

        u_hats_inv = []

        for u in us:
            if u.size == 1:
                inv = jnp.ones_like(u)  # damping not used in the scalar factors

            elif u.ndim == 2:
                inv = psd_inv_cholesky(u, d_hat)

            else:  # diagonal case
                assert u.ndim == 1
                inv = 1.0 / (u + d_hat)

            u_hats_inv.append(inv / c_k)

        return tuple(u_hats_inv)

    def zero_inverse() -> Tuple[Array, ...]:
        # In the special case where for some reason one of the factors is zero, then
        # the inverse is just `damping^-1 * I`, hence we write each factor as
        # `damping^(1/k) * I`.

        c_k = jnp.power(damping, 1.0 / len(arrays))

        u_hats_inv = []

        for u in us:
            if u.ndim == 2:
                inv = jnp.eye(u.shape[0], dtype=u.dtype)

            else:
                inv = jnp.ones_like(u)

            u_hats_inv.append(inv / c_k)

        return tuple(u_hats_inv)

    if get_special_case_zero_inv():
        return lax.cond(jnp.greater(c, 0.0), regular_inverse, zero_inverse)

    else:
        return regular_inverse()


# temporary monkey patch until kfac works for layers with shape 1
kfac_jax.utils.pi_adjusted_kronecker_inverse = pi_adjusted_kronecker_inverse


def batch_size_extractor(batch):
    _, weights = batch
    return weights.shape[0] * weights.shape[1]
