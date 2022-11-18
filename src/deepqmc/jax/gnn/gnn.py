from functools import partial
from typing import Sequence

import haiku as hk
import jax.numpy as jnp

from .graph import Graph, GraphUpdate, MolecularGraphEdgeBuilder
from .utils import NodeEdgeMapping


class MessagePassingLayer(hk.Module):
    r"""
    Base class for all message passing layers.

    Args:
        ilayer (int): the index of the current layer in the list of all layers
        shared (dict): attribute names and values which are shared between the
            layers and the :class:`GraphNeuralNetwork` instance.
    """

    def __init__(self, ilayer, shared):
        super().__init__()
        for k, v in shared.items():
            setattr(self, k, v)
        self.first_layer = ilayer == 0
        self.last_layer = ilayer == self.n_interactions - 1
        self.edge_types = tuple(
            typ
            for typ in self.edge_types
            if not self.last_layer or typ not in {'nn', 'en'}
        )
        self.mapping = NodeEdgeMapping(
            self.edge_types,
            node_data={
                'n_nodes': {'nuclei': self.n_nuc, 'electrons': self.n_up + self.n_down},
                'n_node_types': {
                    'nuclei': self.n_nuc,
                    'electrons': 1 if self.n_up == self.n_down else 2,
                },
            },
        )
        self.update_graph = GraphUpdate(
            update_nodes_fn=self.get_update_nodes_fn(),
            update_edges_fn=self.get_update_edges_fn(),
            aggregate_edges_for_nodes_fn=self.get_aggregate_edges_for_nodes_fn(),
        )

    def __call__(self, graph):
        r"""
        Execute the message passing layer.

        Args:
            graph (:class:`Graph`)

        Returns:
            :class:`Graph`: updated graph
        """
        return self.update_graph(graph)

    def get_update_edges_fn(self):
        r"""
        Create a function that updates the graph edges.

        Returns:
            :data:`Callable[GraphNodes,GraphEdges]`: a function
            that outputs the updated edges as a :class:`GraphEdges` instance.
        """
        raise NotImplementedError

    def get_update_nodes_fn(self):
        r"""
        Create a function that updates the graph nodes.

        Returns:
            :data:`Callable[GraphNodes,*]`: a function
            that outputs the updated nodes as a :class:`GraphNodes` instance.
            The second argument will be the aggregated graph edges.
        """
        raise NotImplementedError

    def get_aggregate_edges_for_nodes_fn(self):
        r"""
        Create a function that aggregates the graph edges.

        Returns:
            :data:`Callable[GraphNodes,GraphEdges]`: a function
            that outputs the aggregated edges.
        """
        raise NotImplementedError


class GraphNeuralNetwork(hk.Module):
    r"""
    Base class for all graph neural networks on molecules.

    Args:
        mol (:class:`deepqmc.jax.Molecule`): the molecule on which the GNN
            is defined
        embedding_dim (int): the size of the electron embeddings to be returned.
        cutoff (float): cutoff distance above which graph edges are discarded.
        n_interactions (int): the number of interaction layers in the GNN.
        layer_kwargs (dict): optional, kwargs to be passed to the layers.
        ghost_coords (float, [N, 3]): optional, coordinates of ghost atoms.
            These will be included as nuclear nodes in the graph. Useful for
            breaking undesired spatial symmetries.
        share_with_layers (dict): optional, attribute names and values to share
            with the interaction layers.
    """

    def __init__(
        self,
        mol,
        embedding_dim,
        cutoff,
        n_interactions,
        layer_kwargs=None,
        ghost_coords=None,
        share_with_layers=None,
    ):
        super().__init__()
        n_nuc, n_up, n_down = mol.n_particles
        self.coords = mol.coords
        self.cutoff = cutoff
        if ghost_coords is not None:
            self.coords = jnp.concatenate([self.coords, jnp.asarray(ghost_coords)])
            n_nuc = len(self.coords)
        share_with_layers = share_with_layers or {}
        share_with_layers.setdefault('embedding_dim', embedding_dim)
        share_with_layers.setdefault('n_nuc', n_nuc)
        share_with_layers.setdefault('n_up', n_up)
        share_with_layers.setdefault('n_down', n_down)
        for k, v in share_with_layers.items():
            setattr(self, k, v)
        share_with_layers.setdefault('edge_types', self.edge_types)
        share_with_layers.setdefault('n_interactions', n_interactions)
        self.layers = [
            self.layer_factory(
                i,
                share_with_layers,
                **(
                    layer_kwargs[i]
                    if isinstance(layer_kwargs, Sequence)
                    else (layer_kwargs or {})
                ),
            )
            for i in range(n_interactions)
        ]
        self.n_up, self.n_down = n_up, n_down

    def init_state(self, shape, dtype):
        r"""Initialize the haiku state that communicates the sizes of edge lists."""
        raise NotImplementedError

    def node_factory(self):
        r"""Return the initial node representations as a :class:`GraphNodes` instance.
        """
        raise NotImplementedError

    def edge_feature_callback(
        self, edge_type, pos_sender, pos_receiver, sender_idx, receiver_idx
    ):
        r"""
        Define the :func:`feature_callback` for the different types of edges.

        Args:
            edge_typ (str): name of the edge_type for which features are calculated.
            pos_sender (float, (:math:`N_\text{nodes}`, 3)): coordinates of the
                sender nodes.
            pos_receiver (float, (:math:`M_\text{nodes}`, 3]): coordinates of the
                receiver nodes.
            sender_idx (int, (:data:`occupancy_limit`)): indeces of the sender nodes.
            receiver_idx (int, (:data:`occupancy_limit`)): indeces of the receiver
                nodes.

        Returns:
            the features for the given edges
        """
        raise NotImplementedError

    @classmethod
    @property
    def edge_types(cls):
        r"""
        Return a tuple containing the names of the edge types used in the GNN.

        See :class:`~deepqmc.jax.gnn.graph.MolecularGraphEdgeBuilder` for possible
        edge types.
        """
        raise NotImplementedError

    def edge_factory(self, r, occupancies):
        r"""Return a function that builds all the edges used in the GNN."""
        edge_factory = MolecularGraphEdgeBuilder(
            self.n_nuc,
            self.n_up,
            self.n_down,
            self.coords,
            self.edge_types,
            kwargs_by_edge_type={
                typ: {
                    'cutoff': self.cutoff[typ],
                    'feature_callback': partial(self.edge_feature_callback, typ),
                }
                for typ in self.edge_types
            },
        )
        return edge_factory(r, occupancies)

    @classmethod
    @property
    def layer_factory(cls):
        r"""Return the class of the interaction layer to be used."""
        return MessagePassingLayer

    def process_final_embedding(self, final_embedding):
        r"""Process final embedding produced by last layer."""
        return final_embedding.electrons['embedding']

    def __call__(self, r):
        r"""
        Execute the graph neural network.

        Args:
            r (float, (:math:`N_\text{elec}`, 3)): electron coordinates.

        Returns:
            float, (:math:`N_\text{elec}`, :data:`embedding_dim`):
            the final embeddings of the electrons.
        """
        occupancies = hk.get_state(
            'occupancies',
            shape=1,
            dtype=jnp.int32,
            init=self.init_state,
        )
        graph_nodes = self.node_factory()
        graph_edges, occupancies = self.edge_factory(r, occupancies)
        hk.set_state('occupancies', occupancies)
        graph = Graph(graph_nodes, graph_edges)

        for layer in self.layers:
            graph = layer(graph)

        return self.process_final_embedding(graph.nodes)
