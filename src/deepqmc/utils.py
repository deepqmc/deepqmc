from __future__ import annotations

from collections.abc import Callable, Iterable, MutableMapping, Sequence
from functools import partial
from typing import Optional, TypeVar

import jax
import jax.numpy as jnp
from jax import ops
from jax.random import uniform
from jax.scipy.special import gammaln
from jaxtyping import PyTree

from .types import Stats

__all__ = ()

T = TypeVar('T')


def flatten(x: jax.Array, start_axis: int = 0) -> jax.Array:
    return x.reshape(*x.shape[:start_axis], -1)


def unflatten(x: jax.Array, axis: int, shape: tuple[int, ...]) -> jax.Array:
    if axis < 0:
        axis = len(x.shape) + axis
    begin = x.shape[:axis]
    end = x.shape[axis + 1 :]
    return x.reshape(*begin, *shape, *end)


def multinomial_resampling(rng, weights, n_samples=None):
    n = len(weights)
    n_samples = n_samples or n
    weights_normalized = weights / jnp.sum(weights)
    i, j = jnp.triu_indices(n)
    weights_cum = jnp.zeros((n, n)).at[i, j].set(weights_normalized[j]).sum(axis=-1)
    return n - 1 - (uniform(rng, (n_samples,))[:, None] > weights_cum).sum(axis=-1)


def factorial2(n):
    n = jnp.asarray(n)
    gamma = jnp.exp(gammaln(n / 2 + 1))
    factor = jnp.where(
        n % 2, jnp.power(2, n / 2 + 0.5) / jnp.sqrt(jnp.pi), jnp.power(2, n / 2)
    )
    return factor * gamma


def masked_mean(x, mask, axis=None):
    x = jnp.where(mask, x, 0)
    assert isinstance(x, jax.Array)
    return x.sum(axis=axis) / jnp.sum(mask, axis=axis)


def triu_flat(x):
    i, j = jnp.triu_indices(x.shape[-1], 1)
    return x[..., i, j]


def tree_norm(x: PyTree) -> float:
    return jax.tree.reduce(lambda norm, x: norm + jnp.linalg.norm(x), x, 0.0)


def tree_stack(trees: list[PyTree]) -> PyTree:
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)


def tree_unstack(tree: PyTree) -> list[PyTree]:
    leaves, treedef = jax.tree.flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves)]


def tree_any(x: PyTree) -> bool:
    return jax.tree.reduce(lambda is_any, leaf: is_any or leaf, x, False)


def norm(rs: jax.Array, safe: bool = False, axis: int = -1) -> jax.Array:
    eps = jnp.finfo(rs.dtype).eps
    return (
        jnp.sqrt(eps + (rs * rs).sum(axis=axis))
        if safe
        else jnp.linalg.norm(rs, axis=axis)
    )


def split_dict(
    dct: dict[str, T], cond: Callable[[str], bool]
) -> tuple[dict[str, T], dict[str, T]]:
    included: dict[str, T] = {}
    excluded: dict[str, T] = {}
    for k, v in dct.items():
        (included if cond(k) else excluded)[k] = v
    return included, excluded


def InverseSchedule(init_value: float, decay_rate: float) -> Callable[[int], float]:
    r"""Create a schedule that decays inversely proportional to the step count.

    Returns a callable computing :math:`f(n) = \text{init\_value} / (1 + n /
    \text{decay\_rate})`. Commonly used as the learning rate or damping schedule
    of :class:`~deepqmc.optimizer.KFACOptimizer`, e.g. via
    ``_target_: deepqmc.utils.InverseSchedule`` in a hydra config.

    Args:
        init_value (float): the value of the schedule at step 0.
        decay_rate (float): the number of steps after which the value has decayed
            to half of :data:`init_value`.

    Returns:
        ~collections.abc.Callable[[int], float]: a function mapping the step
        number to the current schedule value.
    """
    return lambda n: init_value / (1 + n / decay_rate)


def ConstantSchedule(value: float) -> Callable[[int], float]:
    r"""Create a schedule that returns the same value at every step.

    Commonly used as the learning rate or damping schedule of
    :class:`~deepqmc.optimizer.KFACOptimizer`, e.g. via
    ``_target_: deepqmc.utils.ConstantSchedule`` in a hydra config.

    Args:
        value (float): the constant value of the schedule.

    Returns:
        ~collections.abc.Callable[[int], float]: a function mapping the step
        number to :data:`value`.
    """
    return lambda n: value


def argmax_random_choice(rng, x):
    logits = jnp.where(x == x.max(), 0, -jnp.inf)
    return jax.random.categorical(rng, logits, shape=())


def segment_nanmean(
    data: jax.Array, segment_ids: jax.Array, num_segments: int
) -> jax.Array:
    mask = ~jnp.isnan(data)
    counts = jnp.bincount(
        jnp.where(mask, segment_ids, num_segments), length=num_segments
    )
    nanmean = (
        ops.segment_sum(jnp.where(mask, data, 0), segment_ids, num_segments) / counts
    )
    return nanmean


def segment_nanstd(data: jax.Array, segment_ids: jax.Array, num_segments: int):
    mask = ~jnp.isnan(data)
    counts = jnp.bincount(
        jnp.where(mask, segment_ids, num_segments), length=num_segments
    )
    nanmean = segment_nanmean(data, segment_ids, num_segments)
    nanstd = jnp.where(mask, (nanmean[segment_ids] - data) ** 2, 0)
    nanstd = jnp.sqrt(ops.segment_sum(nanstd, segment_ids, num_segments) / counts)
    return nanstd


def per_mol_stats(
    n_mols: int,
    data: jax.Array,
    mol_idx: jax.Array,
    prefix: str,
    mean_only: bool = False,
) -> jax.Array | Stats:
    mean = segment_nanmean(data, mol_idx, n_mols)
    if mean_only:
        return mean
    std = segment_nanstd(data, mol_idx, n_mols)
    mask = ~jnp.isnan(data)
    minimum = ops.segment_min(jnp.where(mask, data, jnp.inf), mol_idx, n_mols)
    maximum = ops.segment_max(jnp.where(mask, data, -jnp.inf), mol_idx, n_mols)
    return {
        f'{prefix}/mean': mean,
        f'{prefix}/std': std,
        f'{prefix}/max': maximum,
        f'{prefix}/min': minimum,
    }


def log_squeeze(x: jax.Array):
    sgn, x = jnp.sign(x), jnp.abs(x)
    return sgn * jnp.log1p((x + 1 / 2 * x**2 + x**3) / (1 + x**2))


def weighted_std(
    x: jax.Array, weights: jax.Array, axis: int | Sequence[int] | None = None
) -> jax.Array:
    mean = jnp.average(x, axis=axis, weights=weights, keepdims=True)
    variance = jnp.average((x - mean) ** 2, axis=axis, weights=weights)
    return jnp.sqrt(variance)


def filter_dict(x: MutableMapping, keys_whitelist: Optional[Iterable[str]]) -> dict:
    x_filtered = (
        {
            key: value
            for key, value in x.items()
            if any(k in key for k in keys_whitelist)
        }
        if keys_whitelist is not None
        else {}
    )
    return x_filtered


def permute_matrix(x, idxs):
    return x[idxs][:, idxs]


def flatten_dict(dictionary, parent_key='', separator='/'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def index(array, idxs):
    return array[idxs]


def better_where(condition, true_val, false_val):
    condition = jnp.expand_dims(
        condition, range(len(condition.shape), len(true_val.shape))
    )
    return jnp.where(condition, true_val, false_val)


def to_tuple(o):
    return tuple([to_tuple(i) for i in o]) if isinstance(o, Iterable) else o


def scaled_normal(key, shape, mean=0, std=1):
    return mean + jax.random.normal(key, shape) * std


def broadcast_pytree_structure(x, y) -> tuple:
    """Recursively broadcast two pytrees at the structure level."""

    class CombinedLeaf:
        """A helper class which jax.tree.map recognizes as a leaf by default."""

        __slots__ = ('x', 'y')

        def __init__(self, x, y):
            self.x = x
            self.y = y

    def broadcast_and_combine_pytree_structure(x, y):
        x_children = jax.tree.structure(x).children()
        y_children = jax.tree.structure(y).children()
        if x_children and y_children:
            return jax.tree.map(
                broadcast_and_combine_pytree_structure,
                x,
                y,
                is_leaf=lambda x: x is None,
            )

        if not y_children:
            return jax.tree.map(lambda x_leaf: CombinedLeaf(x_leaf, y), x)

        if not x_children:
            return jax.tree.map(lambda y_leaf: CombinedLeaf(x, y_leaf), y)

    combined = broadcast_and_combine_pytree_structure(x, y)
    return jax.tree.map(lambda combined_leaf: combined_leaf.x, combined), jax.tree.map(
        lambda combined_leaf: combined_leaf.y, combined
    )


def batched_vmap(func, batch_size: int, in_axes: int | tuple = 0, out_axis: int = 0):
    """A version of jax.vmap that splits the mapped axis into batches of a given size.

    Useful to reduce the memory requirements of vmap.
    """

    def is_none(x) -> bool:
        return x is None

    def arg_size_reducer(acc: int | None, x: int | None):
        if x is None:
            return acc
        if acc is None:
            return x
        assert acc == x, 'All mapped axes must have the same size'
        return acc

    def batch_slicer(i_batch: int, x: jax.Array, axis: int | None) -> jax.Array:
        if axis is None:
            return x
        return jnp.take(
            x,
            jnp.arange(i_batch * batch_size, (i_batch + 1) * batch_size),
            axis=axis,
            unique_indices=True,
            indices_are_sorted=True,
        )

    def mapped_func(*args):
        broadcasted_in_axes, _ = broadcast_pytree_structure(in_axes, args)
        arg_size = jax.tree.reduce(
            arg_size_reducer,
            jax.tree.map(
                lambda axis, x: None if axis is None else x.shape[axis],
                broadcasted_in_axes,
                args,
                is_leaf=is_none,
            ),
            initializer=None,
        )
        assert arg_size is not None, 'At least one argument must be mapped'

        outs = []
        for i_batch in range(arg_size // batch_size):
            args_batch = jax.tree.map(
                partial(batch_slicer, i_batch),
                args,
                broadcasted_in_axes,
                is_leaf=is_none,
            )
            outs.append(jax.vmap(func, in_axes)(*args_batch))
        return jax.tree.map(lambda *x: jnp.concatenate(x, axis=out_axis), *outs)

    return mapped_func
