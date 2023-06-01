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
]


def offdiagonal_sender_idx(n_node):
    return (
        jnp.arange(n_node)[None, :] <= jnp.arange(n_node - 1)[:, None]
    ) + jnp.arange(n_node - 1)[:, None]


def compute_edges(pos_sender, pos_receiver, filter_diagonal):
    diffs = pos_receiver[..., None, :, :] - pos_sender[..., None, :]
    if filter_diagonal:
        assert pos_sender.shape[-2] == pos_receiver.shape[-2]
        n_node = pos_sender.shape[-2]
        receiver_idx = jnp.broadcast_to(jnp.arange(n_node)[None], (n_node - 1, n_node))
        sender_idx = offdiagonal_sender_idx(n_node)
        diffs = diffs[..., sender_idx, receiver_idx, :]
    return diffs


def GraphEdgeBuilder(
    mask_self,
    offsets,
    mask_vals,
):
    r"""
    Create a function that builds graph edges.

    Args:
        filter_self (bool): whether to filter edges between nodes of the same index.
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

        return compute_edges(pos_sender, pos_receiver, mask_self)

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
        lambda xs: jnp.concatenate([x.reshape(-1, x.shape[-1]) for x in xs]),
        edge_of_lists,
        is_leaf=lambda x: isinstance(x, list),
    )


def MolecularGraphEdgeBuilder(n_nuc, n_up, n_down, edge_types):
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
    """
    n_elec = n_up + n_down
    builder_mapping = {
        'nn': ['nn'],
        'ne': ['ne'],
        'en': ['en'],
        'same': ['uu', 'dd'],
        'anti': ['ud', 'du'],
        'up': ['up'],
        'down': ['down'],
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
        'up': {'mask_self': False, 'offsets': (0, 0), 'mask_vals': (n_elec, n_elec)},
        'down': {
            'mask_self': False,
            'offsets': (n_up, 0),
            'mask_vals': (n_elec, n_elec),
        },
    }
    builders = {
        builder_type: GraphEdgeBuilder(
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
        'up': lambda pc: builders['up'](pc.r[:n_up], pc.r),
        'down': lambda pc: builders['down'](pc.r[n_up:], pc.r),
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

        if update_nodes_fn:
            aggregated_edges = aggregate_edges_for_nodes_fn(nodes, edges)
            nodes = update_nodes_fn(nodes, aggregated_edges)

        if update_edges_fn:
            edges = update_edges_fn(edges)

        return Graph(nodes, edges)

    return update_graph
