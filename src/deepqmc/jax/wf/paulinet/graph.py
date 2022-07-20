from functools import partial

import jax.numpy as jnp
import jax.tree_util as tree
from jax import jit, vmap
from jax_md import partition, space
from jraph import GraphsTuple


# Node types: 0: nucleus, 1: spin-up elec., 2: spin-down elec, 3: padding node
@partial(jit, static_argnums=(1, 2, 3))
def type_of_nodes(nodes, n_nuc, n_up, n_down):
    ones = jnp.ones_like(nodes)
    return jnp.where(
        nodes < n_nuc,
        0 * ones,
        jnp.where(
            nodes < n_nuc + n_up,
            1 * ones,
            jnp.where(nodes < n_nuc + n_up + n_down, 2 * ones, 3 * ones),
        ),
    )


# Edge types: 0: nuc->nuc, 1: nuc->e, 2: e->nuc, 3: e->e same, 4: e->e anti,
#             5: padding edge
@partial(jit, static_argnums=(2, 3, 4))
def type_of_edges(senders, receivers, n_nuc, n_up, n_down):
    ones = jnp.ones_like(senders)
    senders_type = type_of_nodes(senders, n_nuc, n_up, n_down)
    receivers_type = type_of_nodes(receivers, n_nuc, n_up, n_down)

    diff_sender_elec = jnp.where(receivers_type == 0, 2 * ones, 4 * ones)
    same = jnp.where(senders_type == 0, 0 * ones, 3 * ones)
    diff = jnp.where(senders_type == 0, ones, diff_sender_elec)
    real = jnp.where(senders_type == receivers_type, same, diff)
    return jnp.where(
        jnp.logical_or(senders_type == 3, receivers_type == 3), 5 * ones, real
    )


class GraphBuilder:
    def __init__(
        self,
        n_nuclei,
        n_up,
        n_down,
        cutoff,
        ref_position,
        distance_basis,
        elec_embeddings,
        nuc_embeddings,
    ):
        displacement, shift = space.free()
        self.metric = space.metric(displacement)

        self.neighbor_fn = partition.neighbor_list(
            displacement, None, cutoff, format=partition.NeighborListFormat.Sparse
        )
        self.neighbors = self.neighbor_fn.allocate(ref_position)
        self.elec_embeddings = elec_embeddings
        self.nuc_embeddings = nuc_embeddings
        self.db = distance_basis
        self.n_nuclei = n_nuclei
        self.n_up = n_up
        self.n_down = n_down
        self.n_particles = n_nuclei + n_up + n_down

    def _update_neighbors(self, positions):
        new_neighbors = vmap(self.neighbors.update)(positions)
        while new_neighbors.did_buffer_overflow.any():
            candidate_neighbors = self.neighbor_fn.allocate(
                positions[new_neighbors.did_buffer_overflow][0]
            )
            new_neighbors = vmap(candidate_neighbors.update)(positions)
        self.neighbors = new_neighbors

    def build(self, positions):
        batch_size = len(positions)
        self._update_neighbors(positions)
        _graph = vmap(partition.to_jraph)(self.neighbors)
        senders = _graph.senders.reshape(-1)
        receivers = _graph.receivers.reshape(-1)
        offset = jnp.tile(
            self.n_particles * jnp.arange(batch_size)[:, None],
            (1, len(senders) // batch_size),
        ).reshape(-1)
        dist_neighbors = vmap(self.metric)(
            positions.reshape(-1, 3)[senders + offset],
            positions.reshape(-1, 3)[receivers + offset],
        )
        nodes = {
            'nuc': jnp.tile(self.nuc_embeddings[None], (batch_size, 1, 1)),
            'elec': jnp.tile(self.elec_embeddings[None], (batch_size, 1, 1)),
        }
        edges = {
            'dist': self.db(dist_neighbors),
            'type': type_of_edges(
                senders, receivers, self.n_nuclei, self.n_up, self.n_down
            ),
        }
        graph = GraphsTuple(
            nodes=nodes,
            edges=edges,
            receivers=receivers + offset,
            senders=senders + offset,
            n_node=self.n_particles * batch_size,
            n_edge=len(senders),
            globals=None,
        )
        return graph


def GraphNetwork(
    aggregate_edges_for_nodes_fn,
    update_node_fn=None,
    update_edge_fn=None,
):
    def _ApplyGraphNet(graph):
        nodes, edges, receivers, senders, globals_, n_node, n_edge = graph

        sent_attributes = tree.tree_map(lambda n: n[senders], nodes)
        received_attributes = tree.tree_map(lambda n: n[receivers], nodes)

        if update_edge_fn:
            edges = update_edge_fn(edges, sent_attributes, received_attributes)

        if update_node_fn:
            received_attributes = aggregate_edges_for_nodes_fn(
                nodes, edges, senders, receivers, n_node
            )
            nodes = update_node_fn(nodes, received_attributes)

        return GraphsTuple(
            nodes=nodes,
            edges=edges,
            receivers=receivers,
            senders=senders,
            globals=globals_,
            n_node=n_node,
            n_edge=n_edge,
        )

    return _ApplyGraphNet
