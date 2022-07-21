from functools import partial

import jax.numpy as jnp
import jax.tree_util as tree
from jax import jit, vmap
from jraph import GraphsTuple

from neighbors import get_neighbors


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
        self.neighbor_fn = get_neighbors
        *_, self.max_occupancy = self.neighbor_fn(ref_position, cutoff)
        self.cutoff = cutoff
        self.elec_embeddings = elec_embeddings
        self.nuc_embeddings = nuc_embeddings
        self.db = distance_basis
        self.n_nuclei = n_nuclei
        self.n_up = n_up
        self.n_down = n_down
        self.n_particles = n_nuclei + n_up + n_down

    def _update_neighbors(self, positions):
        new_neighbors = vmap(self.neighbor_fn, (0, None, None))(
            positions, self.cutoff, self.max_occupancy.item()
        )
        if jnp.any(new_neighbors.did_buffer_overflow):
            new_neighbors = vmap(self.neighbor_fn, (0, None, None))(
                positions, self.cutoff, new_neighbors.max_occupancy.item()
            )
            assert not jnp.any(new_neighbors.did_buffer_overflow)
            self.max_occupancy = new_neighbors.max_occupancy
        return new_neighbors

    def build(self, positions):
        batch_size = len(positions)
        neighbor_list = self._update_neighbors(positions)
        receivers, senders = jnp.split(neighbor_list.idx, 2, axis=1)
        senders = senders.reshape(-1)
        receivers = receivers.reshape(-1)
        offset = jnp.tile(
            self.n_particles * jnp.arange(batch_size)[:, None],
            (1, self.max_occupancy),
        ).reshape(-1)
        nodes = {
            'nuc': jnp.tile(self.nuc_embeddings[None], (batch_size, 1, 1)),
            'elec': jnp.tile(self.elec_embeddings[None], (batch_size, 1, 1)),
        }
        edges = {
            'dist': self.db(neighbor_list.dR.reshape(-1)),
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


# Reproduced from Jraph:
# https://github.com/deepmind/jraph/blob/
# 8b0536c0c8c4fbd8bd197f474add7eec8807c117/jraph/_src/models.py
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
