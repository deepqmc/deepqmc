from collections import namedtuple

import jax.numpy as jnp
import jax_dataclasses as jdc

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
            sender and receiver node indices respectively.
        mask_vals ((int, int)): if ``occupancy_limit`` is larger than the number
            of valid edges, the remaining node indices will be filled with these
            values for the sender and receiver nodes respectively
            (i.e. the value to pad the node index arrays with).
        feature_callback (Callable): a function that takes the sender positions,
            receiver positions, sender node indices and receiver node indices and
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


def MolecularGraphEdgeBuilder(n_nuc, n_up, n_down, edge_types, *, self_interaction):
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
                - ``'same'``: edges between same-spin electrons
                - ``'anti'``: edges between opposite-spin electrons
                - ``'up'``: edges going from spin-up electrons to all electrons
                - ``'down'``: edges going from spin-down electrons to all electrons
        self_interaction (bool): whether edges between a particle and itself are
            considered
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
            'mask_self': not self_interaction,
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
        'uu': {
            'mask_self': not self_interaction,
            'offsets': (0, 0),
            'mask_vals': (n_elec, n_elec),
        },
        'dd': {
            'mask_self': not self_interaction,
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

    build_rules = {
        'nn': lambda pc: SimpleGraphEdges(builders['nn'](pc.R, pc.R)),
        'ne': lambda pc: SimpleGraphEdges(builders['ne'](pc.R, pc.r)),
        'en': lambda pc: SimpleGraphEdges(builders['en'](pc.r, pc.R)),
        'same': lambda pc: SameGraphEdges(
            builders['uu'](pc.r[:n_up], pc.r[:n_up]),
            builders['dd'](pc.r[n_up:], pc.r[n_up:]),
        ),
        'anti': lambda pc: AntiGraphEdges(
            builders['du'](pc.r[n_up:], pc.r[:n_up]),
            builders['ud'](pc.r[:n_up], pc.r[n_up:]),
        ),
        'up': lambda pc: UpGraphEdges(builders['up'](pc.r[:n_up], pc.r)),
        'down': lambda pc: DownGraphEdges(builders['down'](pc.r[n_up:], pc.r)),
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


class GraphEdges:
    @property
    def single_array(self):
        raise NotImplementedError

    def update_from_single_array(self, array):
        raise NotImplementedError

    def sum_senders(self, normalize=False):
        raise NotImplementedError

    def convolve(self, nodes, normalize=False):
        raise NotImplementedError


@jdc.pytree_dataclass
class SimpleGraphEdges(GraphEdges):
    edges: jnp.ndarray

    @property
    def single_array(self):
        return self.edges

    def update_from_single_array(self, array):
        return self.__class__(array)

    def sum_senders(self, normalize=False):
        return (jnp.mean if normalize else jnp.sum)(self.edges, axis=-3)

    def convolve(self, nodes, normalize=False):
        edge_node_product = self.edges * nodes[:, None]
        return self.__class__(edge_node_product).sum_senders(normalize)


@jdc.pytree_dataclass
class UpGraphEdges(SimpleGraphEdges):
    def convolve(self, nodes, normalize=False):
        up = self.edges * nodes[: self.edges.shape[-3], None]
        return self.__class__(up).sum_senders(normalize)


@jdc.pytree_dataclass
class DownGraphEdges(SimpleGraphEdges):
    def convolve(self, nodes, normalize=False):
        down = self.edges * nodes[-self.edges.shape[-3] :, None]
        return self.__class__(down).sum_senders(normalize)


@jdc.pytree_dataclass
class SameGraphEdges(GraphEdges):
    uu: jnp.ndarray
    dd: jnp.ndarray

    @property
    def single_array(self):
        batch_dims = self.uu.shape[:-3]
        return jnp.concatenate(
            [
                self.uu.reshape(*batch_dims, -1, self.uu.shape[-1]),
                self.dd.reshape(*batch_dims, -1, self.dd.shape[-1]),
            ],
            axis=-2,
        )

    def update_from_single_array(self, array):
        n_up = self.uu.shape[-2]
        n_down = self.dd.shape[-2]
        n_sender_up = self.uu.shape[-3]
        n_sender_down = self.dd.shape[-3]
        uu, dd = jnp.split(array, (n_up * n_sender_up,), axis=-2)
        uu = uu.reshape(*uu.shape[:-2], n_sender_up, n_up, uu.shape[-1])
        dd = dd.reshape(*dd.shape[:-2], n_sender_down, n_down, dd.shape[-1])
        return self.__class__(uu, dd)

    def sum_senders(self, normalize=False):
        norm_uu, norm_dd = (
            max(x.shape[-3], 1) if normalize else 1 for x in (self.uu, self.dd)
        )
        up, down = (
            jnp.sum(self.uu, axis=-3) / norm_uu,
            jnp.sum(self.dd, axis=-3) / norm_dd,
        )
        return jnp.concatenate([up, down], axis=-2)

    def convolve(self, nodes, normalize=False):
        self_interaction = self.uu.shape[-3] == self.uu.shape[-2]
        up_node_idx = (
            (slice(None, self.uu.shape[-2]), None)
            if self_interaction
            else offdiagonal_sender_idx(self.uu.shape[-2])
        )
        down_node_idx = (
            (slice(self.uu.shape[-2], None), None)
            if self_interaction
            else self.uu.shape[-2] + offdiagonal_sender_idx(self.dd.shape[-2])
        )
        uu = self.uu * nodes[up_node_idx]
        dd = self.dd * nodes[down_node_idx]
        return self.__class__(uu, dd).sum_senders(normalize)


@jdc.pytree_dataclass
class AntiGraphEdges(GraphEdges):
    du: jnp.ndarray
    ud: jnp.ndarray

    @property
    def single_array(self):
        batch_dims = self.du.shape[:-3]
        return jnp.concatenate(
            [
                self.du.reshape(*batch_dims, -1, self.du.shape[-1]),
                self.ud.reshape(*batch_dims, -1, self.ud.shape[-1]),
            ],
            axis=-2,
        )

    def update_from_single_array(self, array):
        n_up = self.du.shape[-2]
        n_down = self.ud.shape[-2]
        du, ud = jnp.split(array, (n_up * n_down,))
        du = du.reshape(*du.shape[:-2], n_down, n_up, du.shape[-1])
        ud = ud.reshape(*ud.shape[:-2], n_up, n_down, ud.shape[-1])
        return self.__class__(du, ud)

    def sum_senders(self, normalize=False):
        norm_du, norm_ud = (
            max(x.shape[-3], 1) if normalize else 1 for x in (self.du, self.ud)
        )
        up, down = (
            jnp.sum(self.du, axis=-3) / norm_du,
            jnp.sum(self.ud, axis=-3) / norm_ud,
        )
        return jnp.concatenate([up, down], axis=-2)

    def convolve(self, nodes, normalize=False):
        du = self.du * nodes[self.du.shape[-2] :, None]
        ud = self.ud * nodes[: self.du.shape[-2], None]
        return self.__class__(du, ud).sum_senders(normalize)
