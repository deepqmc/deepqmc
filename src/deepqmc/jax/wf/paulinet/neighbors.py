# Adapted from JAX MD:
# https://github.com/google/jax-md/blob/
# e13f48fcf6a18a957ff7d8d46c5664f180cbfcf3/jax_md/partition.py
from collections import namedtuple
from functools import partial

import jax.numpy as jnp
from jax import jit, tree_util, vmap

NeighborList = namedtuple('NeighborList', 'idx dR did_buffer_overflow occupancy')


@jit
def candidate_fn(position):
    candidates = jnp.arange(position.shape[0])
    return jnp.broadcast_to(candidates[None, :], (position.shape[0], position.shape[0]))


@jit
def mask_self_fn(idx):
    self_mask = idx == jnp.reshape(
        jnp.arange(idx.shape[0], dtype=jnp.int32), (idx.shape[0], 1)
    )
    return jnp.where(self_mask, idx.shape[0], idx)


@partial(jit, static_argnames='occupancy_limit')
def prune_neighbor_list_sparse(position, cutoff, idx, occupancy_limit):
    def d(a, b):
        return jnp.sqrt(((b - a) ** 2).sum())

    d = vmap(d)

    N = position.shape[0]
    sender_idx = jnp.broadcast_to(jnp.arange(N)[:, None], idx.shape)

    sender_idx = jnp.reshape(sender_idx, (-1,))
    receiver_idx = jnp.reshape(idx, (-1,))
    dR = d(position[sender_idx], position[receiver_idx])

    mask = (dR < cutoff) & (receiver_idx < N)

    out_idx = N * jnp.ones(occupancy_limit, jnp.int32)
    out_dR = N * jnp.ones(occupancy_limit, jnp.float32)

    cumsum = jnp.cumsum(mask)
    index = jnp.where(mask, cumsum - 1, len(receiver_idx) - 1)
    receiver_idx = out_idx.at[index].set(receiver_idx)
    sender_idx = out_idx.at[index].set(sender_idx)
    dR = out_dR.at[index].set(dR)
    max_occupancy = cumsum[-1]
    idx = idx[:, :occupancy_limit]
    dR = dR[:occupancy_limit]

    return jnp.stack((receiver_idx, sender_idx)), dR, max_occupancy


@partial(jit, static_argnames=('occupancy_limit', 'mask_self'))
def compute_neighbor_list(position, cutoff, occupancy_limit=8, mask_self=True):
    idx = candidate_fn(position)
    if mask_self:
        idx = mask_self_fn(idx)

    idx, dR, occupancy = prune_neighbor_list_sparse(
        position, cutoff, idx, occupancy_limit
    )

    return NeighborList(idx, dR, occupancy > occupancy_limit, occupancy)


class NeighborListBuilder:
    def __init__(self, cutoff, occupancy_limit):
        self.cutoff = cutoff
        self.occupancy_limit = occupancy_limit

    def __call__(self, positions):
        """Creates neighbor list form particle positions.

        Cannot be jitted because shape of neighbor list depends on data.
        We first try to compute the neighbor list with the previously used
        occupancy limit, where we can reuse previously compiled functions.
        If this overflows, because the new positions result in more neighbors,
        we recompile the relevant functions to accomodate larger neighbor
        lists, and redo the calculation with those.
        """
        assert positions.shape[-1] == 3
        assert len(positions.shape) > 1

        batch_dims = positions.shape[:-2]
        _pos = positions.reshape(-1, *positions.shape[-2:])
        neighbor_list = vmap(compute_neighbor_list, (0, None, None))(
            _pos, self.cutoff, self.occupancy_limit
        )
        if jnp.any(neighbor_list.did_buffer_overflow):
            self.occupancy_limit = jnp.max(neighbor_list.occupancy).item()
            neighbor_list = vmap(compute_neighbor_list, (0, None, None))(
                _pos, self.cutoff, self.occupancy_limit
            )
        assert not jnp.any(neighbor_list.did_buffer_overflow)

        return tree_util.tree_map(
            lambda x: x.reshape(*batch_dims, *x.shape[1:]), neighbor_list
        )
