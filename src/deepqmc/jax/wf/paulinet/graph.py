from collections import namedtuple
from functools import partial

import jax.numpy as jnp
from jax import jit, tree_util, vmap

GraphEdges = namedtuple('GraphEdges', 'senders receivers data')
GraphNodes = namedtuple('GraphNodes', 'nuclei electrons')
Graph = namedtuple('Graph', 'nodes edges')

DEFAULT_EDGE_KWARGS = {
    'SchNet': {'cutoff': 10.0, 'occupancy_limit': 2, 'compute_directions': False}
}


def all_graph_edges(pos1, pos2):
    idx = jnp.arange(pos2.shape[0])
    return jnp.broadcast_to(idx[None, :], (pos1.shape[0], pos2.shape[0]))


def mask_self_edges(idx):
    self_mask = idx == jnp.reshape(
        jnp.arange(idx.shape[0], dtype=jnp.int32), (idx.shape[0], 1)
    )
    return jnp.where(self_mask, idx.shape[0], idx)


def prune_graph_edges(
    pos_sender, pos_receiver, cutoff, idx, occupancy_limit, compute_directions
):
    def distance(difference):
        return jnp.sqrt((difference**2).sum())

    dist = vmap(distance)

    N_sender, N_receiver = pos_sender.shape[0], pos_receiver.shape[0]
    sender_idx = jnp.broadcast_to(jnp.arange(N_sender)[:, None], idx.shape)

    sender_idx = jnp.reshape(sender_idx, (-1,))
    receiver_idx = jnp.reshape(idx, (-1,))
    differences = pos_sender[sender_idx] - pos_receiver[receiver_idx]
    distances = dist(differences)
    if compute_directions:
        directions = differences / jnp.where(
            distances[..., None] > jnp.finfo(jnp.float32).eps,
            distances[..., None],
            jnp.finfo(jnp.float32).eps,
        )

    mask = (distances < cutoff) & (receiver_idx < N_receiver)
    cumsum = jnp.cumsum(mask)
    occupancy = cumsum[-1]
    index = jnp.where(mask, cumsum - 1, len(receiver_idx) - 1)

    out_sender_idx = N_sender * jnp.ones(occupancy_limit, jnp.int32)
    out_receiver_idx = N_receiver * jnp.ones(occupancy_limit, jnp.int32)
    out_distances = jnp.zeros(occupancy_limit, jnp.float32)
    if compute_directions:
        out_directions = jnp.zeros((occupancy_limit, 3), jnp.float32)

    receiver_idx = out_receiver_idx.at[index].set(receiver_idx)
    sender_idx = out_sender_idx.at[index].set(sender_idx)
    distances = out_distances.at[index].set(distances)
    if compute_directions:
        directions = out_directions.at[index, :].set(directions)
        return GraphEdges(
            sender_idx,
            receiver_idx,
            {'occupancy': occupancy, 'distances': distances, 'directions': directions},
        )

    return GraphEdges(
        sender_idx, receiver_idx, {'occupancy': occupancy, 'distances': distances}
    )


@partial(jit, static_argnums=(3, 4, 5, 6, 7))
def compute_graph_edges(
    pos1,
    pos2,
    cutoff,
    occupancy_limit,
    mask_self,
    sender_offset,
    receiver_offset,
    compute_directions,
):
    edges_idx = all_graph_edges(pos1, pos2)
    if mask_self:
        edges_idx = mask_self_edges(edges_idx)

    edges = prune_graph_edges(
        pos1, pos2, cutoff, edges_idx, occupancy_limit, compute_directions
    )

    return edges


class GraphEdgesBuilder:
    def __init__(self, cutoff, occupancy_limit, mask_self, compute_directions):
        self.cutoff = cutoff
        self.occupancy_limit = occupancy_limit
        self.mask_self = mask_self
        self.compute_directions = compute_directions
        self.compute_edges_fn = vmap(
            compute_graph_edges, (0, 0, None, None, None, None, None, None)
        )

    def __call__(self, pos1, pos2, sender_offset=0, receiver_offset=0):
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
        _pos1 = pos1.reshape(-1, *pos1.shape[-2:])
        _pos2 = pos2.reshape(-1, *pos2.shape[-2:])

        def compute_edges_fn(occupancy_limit):
            return self.compute_edges_fn(
                _pos1,
                _pos2,
                self.cutoff,
                occupancy_limit,
                self.mask_self,
                sender_offset,
                receiver_offset,
                self.compute_directions,
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
