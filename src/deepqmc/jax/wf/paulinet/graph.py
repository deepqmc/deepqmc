from collections import namedtuple
from functools import partial

import jax.numpy as jnp
from jax import jit, tree_util, vmap, lax

GraphEdges = namedtuple('GraphEdges', 'senders receivers data')
GraphNodes = namedtuple('GraphNodes', 'nuclei electrons')
Graph = namedtuple('Graph', 'nodes edges')

DEFAULT_EDGE_KWARGS = {'SchNet': {'cutoff': 10.0, 'occupancy_limit': 2}}


def all_graph_edges(pos1, pos2):
    idx = jnp.arange(pos2.shape[0])
    return jnp.broadcast_to(idx[None, :], (pos1.shape[0], pos2.shape[0]))


def mask_self_edges(idx):
    self_mask = idx == jnp.reshape(
        jnp.arange(idx.shape[0], dtype=jnp.int32), (idx.shape[0], 1)
    )
    return jnp.where(self_mask, idx.shape[0], idx)


def prune_graph_edges(
    pos_sender,
    pos_receiver,
    cutoff,
    idx,
    occupancy_limit,
    send_offset,
    rec_offset,
    send_mask_val,
    rec_mask_val,
):
    def distance(sender, receiver):
        return jnp.sqrt(((receiver - sender) ** 2).sum())

    dist = vmap(distance)

    N_sender, N_receiver = pos_sender.shape[0], pos_receiver.shape[0]
    sender_idx = jnp.broadcast_to(jnp.arange(N_sender)[:, None], idx.shape)

    sender_idx = jnp.reshape(sender_idx, (-1,))
    receiver_idx = jnp.reshape(idx, (-1,))
    distances = dist(pos_sender[sender_idx], pos_receiver[receiver_idx])

    mask = (distances < cutoff) & (receiver_idx < N_receiver)
    cumsum = jnp.cumsum(mask)
    occupancy = cumsum[-1]

    # edge buffer is one larger than occupancy_limit:
    # masked edges assigned to last position and discarded
    out_sender_idx = (send_mask_val - send_offset) * jnp.ones(
        occupancy_limit + 1, jnp.int32
    )
    out_receiver_idx = (rec_mask_val - rec_offset) * jnp.ones(
        occupancy_limit + 1, jnp.int32
    )
    index = jnp.where(mask, cumsum - 1, occupancy_limit)

    sender_idx = (
        out_sender_idx.at[index].set(sender_idx)[:occupancy_limit] + send_offset
    )
    receiver_idx = (
        out_receiver_idx.at[index].set(receiver_idx)[:occupancy_limit] + rec_offset
    )

    return GraphEdges(sender_idx, receiver_idx, {'occupancy': occupancy})


@partial(jit, static_argnums=(3, 4, 5, 6, 7, 8))
def compute_graph_edges(
    pos1,
    pos2,
    cutoff,
    occupancy_limit,
    mask_self,
    send_offset,
    rec_offset,
    send_mask_val,
    rec_mask_val,
):
    edges_idx = all_graph_edges(pos1, pos2)
    if mask_self:
        edges_idx = mask_self_edges(edges_idx)

    edges = prune_graph_edges(
        pos1,
        pos2,
        cutoff,
        edges_idx,
        occupancy_limit,
        send_offset,
        rec_offset,
        send_mask_val,
        rec_mask_val,
    )

    return edges


class GraphEdgesBuilder:
    def __init__(self, cutoff, occupancy_limit, mask_self, send_mask_val, rec_mask_val):
        self.cutoff = cutoff
        self.occupancy_limit = occupancy_limit
        self.mask_self = mask_self
        self.send_mask_val = send_mask_val
        self.rec_mask_val = rec_mask_val
        self.compute_edges_fn = vmap(
            compute_graph_edges, (0, 0, None, None, None, None, None, None, None)
        )

    def __call__(self, pos1, pos2, send_offset=0, rec_offset=0):
        """Creates sparse graph edges form particle positions.

        Cannot be jitted because shape of graph edges depends on data.
        We first try to compute the graph edges with the previously used
        occupancy limit, where we can reuse previously compiled functions.
        If this overflows, because the new positions result in more edges,
        we recompile the relevant functions to accomodate more edges,
        and redo the calculation.
        """
        assert pos1.shape[-1] == 3 and pos2.shape[-1] == 3
        assert len(pos1.shape) > 1
        assert pos1.shape[:-2] == pos2.shape[:-2]
        assert not self.mask_self or pos1.shape[-2] == pos2.shape[-2]

        batch_dims = pos1.shape[:-2]
        _pos1 = lax.stop_gradient(pos1.reshape(-1, *pos1.shape[-2:]))
        _pos2 = lax.stop_gradient(pos2.reshape(-1, *pos2.shape[-2:]))

        def compute_edges_fn(occupancy_limit):
            return self.compute_edges_fn(
                _pos1,
                _pos2,
                self.cutoff,
                occupancy_limit,
                self.mask_self,
                send_offset,
                rec_offset,
                self.send_mask_val,
                self.rec_mask_val,
            )

        edges = compute_edges_fn(self.occupancy_limit)
        max_occupancy = jnp.max(edges.data['occupancy']).item()
        if max_occupancy > self.occupancy_limit:
            self.occupancy_limit = max_occupancy
            edges = compute_edges_fn(self.occupancy_limit)
        del edges.data['occupancy']

        return tree_util.tree_map(lambda x: x.reshape(*batch_dims, *x.shape[1:]), edges)


def GraphUpdate(
    aggregate_edges_for_nodes_fn,
    update_nodes_fn=None,
    update_edges_fn=None,
):
    def _update(graph):
        nodes, edges = graph

        if update_edges_fn:
            edges = update_edges_fn(nodes, edges)

        if update_nodes_fn:
            aggregated_edges = aggregate_edges_for_nodes_fn(nodes, edges)
            nodes = update_nodes_fn(nodes, aggregated_edges)

        return Graph(nodes, edges)

    return _update
