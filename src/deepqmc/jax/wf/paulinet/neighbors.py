# Reproduced from JAX MD:
# https://github.com/google/jax-md/blob/
# e13f48fcf6a18a957ff7d8d46c5664f180cbfcf3/jax_md/partition.py
from collections import namedtuple
from functools import partial

import jax.numpy as jnp
from jax import jit, vmap

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


@jit
def prune_neighbor_list_sparse(position, cutoff, idx):
    def d(a, b):
        return jnp.sqrt(((b - a) ** 2).sum())

    d = vmap(d)

    N = position.shape[0]
    sender_idx = jnp.broadcast_to(jnp.arange(N)[:, None], idx.shape)

    sender_idx = jnp.reshape(sender_idx, (-1,))
    receiver_idx = jnp.reshape(idx, (-1,))
    dR = d(position[sender_idx], position[receiver_idx])

    mask = (dR < cutoff) & (receiver_idx < N)

    out_idx = N * jnp.ones(receiver_idx.shape, jnp.int32)
    out_dR = N * jnp.ones(receiver_idx.shape, jnp.float32)

    cumsum = jnp.cumsum(mask)
    index = jnp.where(mask, cumsum - 1, len(receiver_idx) - 1)
    receiver_idx = out_idx.at[index].set(receiver_idx)
    sender_idx = out_idx.at[index].set(sender_idx)
    dR = out_dR.at[index].set(dR)
    max_occupancy = cumsum[-1]

    return jnp.stack((receiver_idx, sender_idx)), dR, max_occupancy


@partial(jit, static_argnames=('max_occupancy', 'mask_self'))
def get_neighbors(position, cutoff, max_occupancy=8, mask_self=True):
    idx = candidate_fn(position)
    if mask_self:
        idx = mask_self_fn(idx)

    idx, dR, occupancy = prune_neighbor_list_sparse(position, cutoff, idx)
    idx = idx[:, :max_occupancy]
    dR = dR[:max_occupancy]

    return NeighborList(idx, dR, occupancy > max_occupancy, occupancy)
