from collections import namedtuple

import jax.numpy as jnp
from jax.tree_util import tree_map, tree_structure, tree_transpose

from ..utils import no_grad

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


def prune_graph_edges(
    pos_sender,
    pos_receiver,
    cutoff,
    idx,
    occupancy_limit,
    offsets,
    mask_vals,
    feature_callback,
):
    r"""Discards graph edges which have a distance larger than :data:`cutoff`.

    Args:
        pos_sender (float, (:math:`N_{nodes}`, 3)): coordinates of graph nodes
            that send edges.
        pos_receiver (float, (:math:`M_{nodes}`, 3)): coordinates of graph nodes
            that receive edges.
        cutoff (float): cutoff distance above which edges are discarded.
        idx (int, (:math:`N_\text{nodes}`, :math:`N_\text{nodes}`)): matrix of
            receiving node indices as created by :func:`all_graph_edges`
            (or :func:`mask_self_edges`).
        occupancy_limit (int): the number of edges that can be considered
            without overflow. The arrays describing the edges will have
            a last dimension of size :data:`occupancy_limit`.
        offsets ((int, int)): node index offset to be added to the returned
            sender and receiver node indeces respectively.
        mask_vals ((int, int)): if :data:`occupancy_limit` is larger than the number
            of valid edges, the remaining node indices will be filled with these
            values for the sender and receiver nodes respectively
            (i.e. the value to pad the node index arrays with).
        feature_callback (Callable): a function that takes the sender positions,
            receiver positions, sender node indeces and receiver node indeces and
            returns some data (features) computed for the edges.

    Returns:
        ~jax.types.GraphEdges: object containing the indeces of the edge
        sending and edge receiving nodes, along with the features associated
        with the edges.
    """

    def apply_callback(pos_sender, pos2, sender_idx, receiver_idx):
        r"""Apply the feature_callback function, or return no features."""
        return (
            feature_callback(pos_sender, pos2, sender_idx, receiver_idx)
            if feature_callback
            else {}
        )

    if pos_sender.shape[0] == 0 or pos_receiver.shape[0] == 0:
        ones = jnp.ones(occupancy_limit, idx.dtype)
        sender_idx = offsets[0] * ones
        receiver_idx = offsets[1] * ones
        return (
            GraphEdges(
                sender_idx,
                receiver_idx,
                apply_callback(pos_sender, pos_receiver, sender_idx, receiver_idx),
            ),
            jnp.array(0),
        )

    @no_grad
    def dist(sender, receiver):
        r"""Compute pairwise distances between inputs."""
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

    features = apply_callback(pos_sender, pos_receiver, sender_idx, receiver_idx)

    return (
        GraphEdges(sender_idx + offsets[0], receiver_idx + offsets[1], features),
        occupancy,
    )


def difference_callback(pos_sender, pos_receiver, sender_idx, receiver_idx):
    r"""feature_callback computing the Euclidian difference vector for each edge."""
    if len(pos_sender) == 0 or len(pos_receiver) == 0:
        return jnp.zeros((len(sender_idx), 3))
    diffs = pos_receiver[receiver_idx] - pos_sender[sender_idx]
    return diffs


def GraphEdgeBuilder(
    cutoff,
    mask_self,
    offsets,
    mask_vals,
    feature_callback,
):
    r"""
    Create a function that builds graph edges.

    Args:
        cutoff (float): the cutoff distance above which edges are discarded.
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

    def build(pos_sender, pos_receiver, occupancies):
        r"""
        Build graph edges.

        Args:
            pos_sender (float, (:math:`N_{nodes}`, 3)): coordinates of graph nodes
                that send edges.
            pos_receiver (float, (:math:`M_{nodes}`, 3)): coordinates of graph nodes
                that receive edges.
            occupancies (int, (:data:`occupancy_limit`)): array to store
                occupancies in.

        Returns:
            tuple: a tuple containing the graph edges, the input occupancies
            updated with the current occupancy, and the number of stored
            occupancies.
        """
        assert pos_sender.shape[-1] == 3 and pos_receiver.shape[-1] == 3
        assert len(pos_sender.shape) == 2
        assert not mask_self or pos_sender.shape[0] == pos_receiver.shape[0]

        occupancy_limit = occupancies.shape[0]

        edges_idx = all_graph_edges(pos_sender, pos_receiver)

        if mask_self:
            edges_idx = mask_self_edges(edges_idx)
        edges, occupancy = prune_graph_edges(
            pos_sender,
            pos_receiver,
            cutoff,
            edges_idx,
            occupancy_limit,
            offsets,
            mask_vals,
            feature_callback,
        )

        return (
            edges,
            occupancies.at[1:].set(occupancies[:-1]).at[0].set(occupancy),
        )

    return build


def concatenate_edges(edges_and_occs):
    r"""
    Concatenate two edge lists.

    Utility function used only internally, e.g. to concatenate ``uu`` and ``dd``
    edges to get all ``same`` edges.
    """
    edges = [edge_occ[0] for edge_occ in edges_and_occs]
    occupancies = tuple(edge_occ[1] for edge_occ in edges_and_occs)
    edge_of_lists = tree_transpose(
        tree_structure([0] * len(edges)), tree_structure(edges[0]), edges
    )
    return (
        tree_map(jnp.concatenate, edge_of_lists, is_leaf=lambda x: isinstance(x, list)),
        occupancies,
    )


def MolecularGraphEdgeBuilder(
    n_nuc, n_up, n_down, nuc_coords, edge_types, kwargs_by_edge_type=None
):
    r"""
    Create a function that builds many types of molecular edges.

    Args:
        n_nuc (int): number of nuclei.
        n_up (int): number of spin-up electrons.
        n_down (int): number of spin-down electrons.
        nuc_coords (float, (:math:`N_\text{nuc}`, 3)): coordinates of nuclei.
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

    def build_same(r, occs):
        return concatenate_edges(
            [
                builders['uu'](r[:n_up], r[:n_up], occs['same'][0]),
                builders['dd'](r[n_up:], r[n_up:], occs['same'][1]),
            ]
        )

    def build_anti(r, occs):
        return concatenate_edges(
            [
                builders['ud'](r[:n_up], r[n_up:], occs['anti'][0]),
                builders['du'](r[n_up:], r[:n_up], occs['anti'][1]),
            ]
        )

    build_rules = {
        'nn': lambda r, occs: builders['nn'](nuc_coords, nuc_coords, occs['nn']),
        'ne': lambda r, occs: builders['ne'](nuc_coords, r, occs['ne']),
        'en': lambda r, occs: builders['en'](r, nuc_coords, occs['en']),
        'same': build_same,
        'anti': build_anti,
    }

    def build(r, occupancies):
        r"""
        Build many types of molecular graph edges.

        Args:
            r (float (:math:`N_\text{elec}`, 3)): electron coordinates.
            occupancies (dict): mapping of edge type names to arrays where the occupancy
                of the given edge type is stored.
        """
        assert r.shape[0] == n_up + n_down

        edges_and_occs = {
            edge_type: build_rules[edge_type](r, occupancies)
            for edge_type in edge_types
        }
        edges = {k: edge_and_occ[0] for k, edge_and_occ in edges_and_occs.items()}
        occupancies = {k: edge_and_occ[1] for k, edge_and_occ in edges_and_occs.items()}
        return edges, occupancies

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
