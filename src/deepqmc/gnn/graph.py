from collections import namedtuple

import jax.numpy as jnp
from jax.tree_util import tree_map, tree_structure, tree_transpose

GraphEdges = namedtuple('GraphEdges', 'senders receivers features')
GraphNodes = namedtuple('GraphNodes', 'nuclei electrons')
Graph = namedtuple('Graph', 'nodes edges')

__all__ = [
    'GraphEdgeBuilder',
    'MolecularGraphEdgeBuilder',
    'GraphUpdate',
    'difference_callback',
]


def all_graph_edges(pos_sender, pos_receiver):
    r"""Create all graph edges.

    Args:
        pos_sender (float, (:math:`N_\text{nodes}`, 3)): coordinates of graph
            nodes that send edges.
        pos_receiver (float, (:math:`M_\text{nodes}`, 3)): coordinates of graph
            nodes that receive edges.

    Returns:
        int, (:math:`N`, :math:`M`): matrix of node indeces, indicating
        the receiver node of the :math:`N \cdot M` possible edges.
    """
    idx = jnp.arange(pos_receiver.shape[0])
    return jnp.broadcast_to(idx[None, :], (pos_sender.shape[0], pos_receiver.shape[0]))


def mask_self_edges(idx):
    r"""Mask the edges where sender and receiver nodes have the same index.

    Args:
        idx (int, (:math:`N_\text{nodes}`, :math:`N_\text{nodes}`)):
            index of receiving nodes, assumed to be square since sets of sender
            and receiver nodes should be identical.

    Returns:
        int, (:math:`N_\text{nodes}`, :math:`N_\text{nodes}`): matrix of
        receiving node indeces, the appropriate entries masked with
        :math:`N_\text{nodes}`.
    """
    self_mask = idx == jnp.reshape(
        jnp.arange(idx.shape[0], dtype=jnp.int32), (idx.shape[0], 1)
    )
    return jnp.where(self_mask, idx.shape[0], idx)


def difference_callback(pos_sender, pos_receiver, sender_idx, receiver_idx):
    r"""feature_callback computing the Euclidian difference vector for each edge."""
    if len(pos_sender) == 0 or len(pos_receiver) == 0:
        return jnp.zeros((len(sender_idx), 3))
    diffs = pos_receiver[receiver_idx] - pos_sender[sender_idx]
    return diffs


def GraphEdgeBuilder(
    mask_self,
    offsets,
    mask_vals,
    feature_callback,
):
    r"""
    Create a function that builds graph edges.

    Args:
        mask_self (bool): whether to mask edges between nodes of the same index.
        offsets ((int, int)): node index offset to be added to the returned
            sender and receiver node indeces respectively.
        mask_vals ((int, int)): if ``occupancy_limit`` is larger than the number
            of valid edges, the remaining node indices will be filled with these
            values for the sender and receiver nodes respectively
            (i.e. the value to pad the node index arrays with).
        feature_callback (Callable): a function that takes the sender positions,
            receiver positions, sender node indeces and receiver node indeces and
            returns some data (features) computed for the edges.
    """

    def build(pos_sender, pos_receiver):
        r"""
        Build graph edges.

        Args:
            pos_sender (float, (:math:`N_{nodes}`, 3)): coordinates of graph nodes
                that send edges.
            pos_receiver (float, (:math:`M_{nodes}`, 3)): coordinates of graph nodes
                that receive edges.

        Returns:
            A :class:`~deepqmc.gnn.graph.GraphEdges` instance.
        """
        assert pos_sender.shape[-1] == 3 and pos_receiver.shape[-1] == 3
        assert len(pos_sender.shape) == 2
        assert not mask_self or pos_sender.shape[0] == pos_receiver.shape[0]

        N_sender, N_receiver = pos_sender.shape[0], pos_receiver.shape[0]

        edges_idx = all_graph_edges(pos_sender, pos_receiver)

        if mask_self:
            edges_idx = mask_self_edges(edges_idx)

        sender_idx = jnp.broadcast_to(jnp.arange(N_sender)[:, None], edges_idx.shape)
        sender_idx = jnp.reshape(sender_idx, (-1,))
        receiver_idx = jnp.reshape(edges_idx, (-1,))

        if mask_self:
            N_edge = N_sender * (N_sender - 1)
            mask = receiver_idx < N_receiver
            cumsum = jnp.cumsum(mask)
            index = jnp.where(mask, cumsum - 1, N_edge)

            # edge buffer is one larger than number of edges:
            # masked edges assigned to last position and discarded
            sender_idx_buf, receiver_idx_buf = (
                (mask_val - offset) * jnp.ones(N_edge + 1, jnp.int32)
                for mask_val, offset in zip(mask_vals, offsets)
            )
            sender_idx = sender_idx_buf.at[index].set(sender_idx)[:N_edge]
            receiver_idx = receiver_idx_buf.at[index].set(receiver_idx)[:N_edge]

        features = feature_callback(pos_sender, pos_receiver, sender_idx, receiver_idx)

        return GraphEdges(sender_idx + offsets[0], receiver_idx + offsets[1], features)

    return build


def concatenate_edges(edges):
    r"""
    Concatenate two edge lists.

    Utility function used only internally, e.g. to concatenate ``uu`` and ``dd``
    edges to get all ``same`` edges.
    """
    edge_of_lists = tree_transpose(
        tree_structure([0] * len(edges)), tree_structure(edges[0]), edges
    )
    return tree_map(
        jnp.concatenate, edge_of_lists, is_leaf=lambda x: isinstance(x, list)
    )


def MolecularGraphEdgeBuilder(
    n_nuc, n_up, n_down, edge_types, kwargs_by_edge_type=None
):
    r"""
    Create a function that builds many types of molecular edges.

    Args:
        n_nuc (int): number of nuclei.
        n_up (int): number of spin-up electrons.
        n_down (int): number of spin-down electrons.
        edge_types (List[str]): list of edge type names to build. Possible names are:

                - ``'nn'``: nuclei->nuclei edges
                - ``'ne'``: nuclei->electrons edges
                - ``'en'``: electrons->nuclei edges
                - ``'same'``: edges betwen same-spin electrons
                - ``'anti'``: edges betwen opposite-spin electrons
        kwargs_by_edge_type (dict): a mapping from names of edge types to the
            kwargs to be passed to the :class:`GraphEdgeBuilder` of that edge type.
    """
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

    def build_same(phys_conf):
        r = phys_conf.r
        return concatenate_edges(
            [
                builders['uu'](r[:n_up], r[:n_up]),
                builders['dd'](r[n_up:], r[n_up:]),
            ]
        )

    def build_anti(phys_conf):
        r = phys_conf.r
        return concatenate_edges(
            [
                builders['ud'](r[:n_up], r[n_up:]),
                builders['du'](r[n_up:], r[:n_up]),
            ]
        )

    build_rules = {
        'nn': lambda pc: builders['nn'](pc.R, pc.R),
        'ne': lambda pc: builders['ne'](pc.R, pc.r),
        'en': lambda pc: builders['en'](pc.r, pc.R),
        'same': build_same,
        'anti': build_anti,
    }

    def build(phys_conf):
        r"""
        Build many types of molecular graph edges.

        Args:
            phys_conf (~deepqmc.types.PhysicalConfiguration): the physical
                configuration of the molecule.
            occupancies (dict): mapping of edge type names to arrays where the occupancy
                of the given edge type is stored.
        """
        assert phys_conf.r.shape[0] == n_up + n_down

        edges = {
            edge_type: build_rules[edge_type](phys_conf) for edge_type in edge_types
        }
        return edges

    return build


def GraphUpdate(
    aggregate_edges_for_nodes_fn,
    update_nodes_fn=None,
    update_edges_fn=None,
):
    r"""
    Create a function that updates a graph.

    The update function is tailored to be used in GNNs.

    Args:
        aggregate_edges_for_nodes_fn (bool): whether to perform the aggregation
            of edges for nodes.
        update_nodes_fn (Callable): optional, function that updates the nodes.
        update_edges_fn (Callable): optional, function that updates the edges.
    """

    def update_graph(graph):
        nodes, edges = graph

        if update_edges_fn:
            edges = update_edges_fn(edges)

        if update_nodes_fn:
            aggregated_edges = aggregate_edges_for_nodes_fn(nodes, edges)
            nodes = update_nodes_fn(nodes, aggregated_edges)

        return Graph(nodes, edges)

    return update_graph
