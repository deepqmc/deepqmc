from collections import namedtuple

import haiku as hk
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_structure, tree_transpose

GraphEdges = namedtuple('GraphEdges', 'senders receivers data')
GraphNodes = namedtuple('GraphNodes', 'nuclei electrons')
Graph = namedtuple('Graph', 'nodes edges')


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
    offsets,
    mask_vals,
    data_callback,
):
    def dist(sender, receiver):
        return jnp.sqrt(((receiver - sender) ** 2).sum(axis=-1))

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
    out_sender_idx, out_receiver_idx = (
        (mask_val - offset) * jnp.ones(occupancy_limit + 1, jnp.int32)
        for mask_val, offset in zip(mask_vals, offsets)
    )
    index = jnp.where(mask, cumsum - 1, occupancy_limit)

    sender_idx = out_sender_idx.at[index].set(sender_idx)[:occupancy_limit]
    receiver_idx = out_receiver_idx.at[index].set(receiver_idx)[:occupancy_limit]

    data = (
        data_callback(pos_sender, pos_receiver, sender_idx, receiver_idx)
        if data_callback
        else {}
    )

    return (
        GraphEdges(sender_idx + offsets[0], receiver_idx + offsets[1], data),
        occupancy,
    )


def distance_callback(pos_sender, pos_receiver, sender_idx, receiver_idx):
    if len(pos_sender) == 0 or len(pos_receiver) == 0:
        return {'distances': jnp.zeros_like(sender_idx, pos_sender.dtype)}
    distances = jnp.sqrt(
        ((pos_receiver[receiver_idx] - pos_sender[sender_idx]) ** 2).sum(axis=-1)
    )
    return {'distances': distances}


def distance_difference_callback(pos_sender, pos_receiver, sender_idx, receiver_idx):
    if len(pos_sender) == 0 or len(pos_receiver) == 0:
        return {
            'distances': jnp.zeros_like(sender_idx, pos_sender.dtype),
            'differences': jnp.zeros((len(sender_idx), 3)),
        }
    differences = pos_receiver[receiver_idx] - pos_sender[sender_idx]
    distances = jnp.sqrt((differences**2).sum(axis=-1))
    return {'distances': distances, 'differences': differences}


def distance_direction_callback(pos_sender, pos_receiver, sender_idx, receiver_idx):
    if len(pos_sender) == 0 or len(pos_receiver) == 0:
        return {
            'distances': jnp.zeros_like(sender_idx, pos_sender.dtype),
            'directions': jnp.zeros((len(sender_idx), 3)),
        }
    eps = jnp.finfo(pos_sender.dtype).eps
    differences = pos_receiver[receiver_idx] - pos_sender[sender_idx]
    distances = jnp.sqrt((differences**2).sum(axis=-1))
    directions = differences / jnp.where(
        distances[..., None] > eps, distances[..., None], eps
    )
    return {'distances': distances, 'directions': directions}


def GraphEdgeBuilder(
    cutoff,
    mask_self,
    offsets,
    mask_vals,
    data_callback=None,
):
    def build(pos1, pos2, occupancies, n_occupancies):
        assert pos1.shape[-1] == 3 and pos2.shape[-1] == 3
        assert len(pos1.shape) == 2
        assert not mask_self or pos1.shape[0] == pos2.shape[0]

        occupancy_limit = occupancies.shape[0]
        if pos1.shape[0] == 0 or pos2.shape[0] == 0:
            ones = jnp.ones_like(occupancies)
            sender_idx = offsets[0] * ones
            receiver_idx = offsets[1] * ones
            return (
                GraphEdges(
                    sender_idx,
                    receiver_idx,
                    data_callback(pos1, pos2, sender_idx, receiver_idx)
                    if data_callback
                    else {},
                ),
                occupancies.at[1:].set(occupancies[:-1]).at[0].set(0),
                n_occupancies + 1,
            )

        edges_idx = all_graph_edges(pos1, pos2)
        if mask_self:
            edges_idx = mask_self_edges(edges_idx)

        edges, occupancy = prune_graph_edges(
            pos1,
            pos2,
            cutoff,
            edges_idx,
            occupancy_limit,
            offsets,
            mask_vals,
            data_callback,
        )
        return (
            edges,
            occupancies.at[1:].set(occupancies[:-1]).at[0].set(occupancy),
            n_occupancies + 1,
        )

    return build


def concatenate_edges(edges_and_occs):
    edges = [edge_occ[0] for edge_occ in edges_and_occs]
    occupancies = tuple(edge_occ[1] for edge_occ in edges_and_occs)
    n_occupancies = edges_and_occs[0][2]
    edge_of_lists = tree_transpose(
        tree_structure([0] * len(edges)), tree_structure(edges[0]), edges
    )
    return (
        tree_map(jnp.concatenate, edge_of_lists, is_leaf=lambda x: isinstance(x, list)),
        occupancies,
        n_occupancies,
    )


def MolecularGraphEdgeBuilder(
    n_nuc, n_up, n_down, coords, edge_types, kwargs_by_edge_type=None
):
    n_elec = n_up + n_down
    builder_mapping = {
        'nn': ['nn'],
        'ne': ['ne'],
        'en': ['en'],
        'same': ['uu', 'dd'],
        'anti': ['ud', 'du'],
    }
    fix_kwargs_of_builder_type = {
        'nn': {
            'mask_self': True,
            'offsets': (0, 0),
            'mask_vals': (n_nuc, n_nuc),
        },
        'ne': {
            'mask_self': False,
            'offsets': (0, 0),
            'mask_vals': (n_nuc, n_elec),
        },
        'en': {
            'mask_self': False,
            'offsets': (0, 0),
            'mask_vals': (n_elec, n_nuc),
        },
        'uu': {'mask_self': True, 'offsets': (0, 0), 'mask_vals': (n_elec, n_elec)},
        'dd': {
            'mask_self': True,
            'offsets': (n_up, n_up),
            'mask_vals': (n_elec, n_elec),
        },
        'ud': {
            'mask_self': False,
            'mask_vals': (n_elec, n_elec),
            'offsets': (0, n_up),
        },
        'du': {
            'mask_self': False,
            'mask_vals': (n_elec, n_elec),
            'offsets': (n_up, 0),
        },
    }
    builders = {
        builder_type: GraphEdgeBuilder(
            **((kwargs_by_edge_type or {}).get(edge_type)),
            **fix_kwargs_of_builder_type[builder_type],
        )
        for edge_type in edge_types
        for builder_type in builder_mapping[edge_type]
    }

    def build_same(r, occs, n_occs):
        return concatenate_edges(
            [
                builders['uu'](r[:n_up], r[:n_up], occs['same'][0], n_occs),
                builders['dd'](r[n_up:], r[n_up:], occs['same'][1], n_occs),
            ]
        )

    def build_anti(r, occs, n_occs):
        return concatenate_edges(
            [
                builders['ud'](r[:n_up], r[n_up:], occs['anti'][0], n_occs),
                builders['du'](r[n_up:], r[:n_up], occs['anti'][1], n_occs),
            ]
        )

    build_rules = {
        'nn': lambda r, occs, n_occs: builders['nn'](
            coords, coords, occs['nn'], n_occs
        ),
        'ne': lambda r, occs, n_occs: builders['ne'](coords, r, occs['ne'], n_occs),
        'en': lambda r, occs, n_occs: builders['en'](r, coords, occs['en'], n_occs),
        'same': build_same,
        'anti': build_anti,
    }

    def build(r, occupancies, n_occupancies):
        assert r.shape[0] == n_up + n_down

        edges_and_occs = {
            edge_type: build_rules[edge_type](r, occupancies, n_occupancies)
            for edge_type in edge_types
        }
        edges = {k: edge_and_occ[0] for k, edge_and_occ in edges_and_occs.items()}
        occupancies = {k: edge_and_occ[1] for k, edge_and_occ in edges_and_occs.items()}
        n_occupancies = next(iter(edges_and_occs.values()))[2]
        return edges, occupancies, n_occupancies

    return build


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


class MessagePassingLayer(hk.Module):
    def __init__(self, name, ilayer):
        super().__init__(f'{name}_{ilayer}')
        self.ilayer = ilayer
        self.update_graph = GraphUpdate(
            update_nodes_fn=self.get_update_nodes_fn(),
            update_edges_fn=self.get_update_edges_fn(),
            aggregate_edges_for_nodes_fn=self.get_aggregate_edges_for_nodes_fn(),
        )

    def __call__(self, graph):
        return self.update_graph(graph)

    def get_update_edges_fn(self):
        pass

    def get_update_nodes_fn(self):
        pass

    def get_aggregate_edges_for_nodes_fn(self):
        pass
