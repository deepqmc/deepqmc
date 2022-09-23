from functools import partial

import haiku as hk
import jax.numpy as jnp

from ..types import Graph, GraphNodes
from .graph import GraphUpdate, MolecularGraphEdgeBuilder


class MessagePassingLayer(hk.Module):
    def __init__(self, ilayer, shared):
        super().__init__()
        self.ilayer = ilayer
        for k, v in shared.items():
            setattr(self, k, v)
        self.update_graph = GraphUpdate(
            update_nodes_fn=self.get_update_nodes_fn(),
            update_edges_fn=self.get_update_edges_fn(),
            aggregate_edges_for_nodes_fn=self.get_aggregate_edges_for_nodes_fn(),
        )

    def __call__(self, graph):
        return self.update_graph(graph)

    def get_update_edges_fn(self):
        raise NotImplementedError

    def get_update_nodes_fn(self):
        raise NotImplementedError

    def get_aggregate_edges_for_nodes_fn(self):
        raise NotImplementedError


class GraphNeuralNetwork(hk.Module):
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
        self.layers = [
            self.layer_factory(
                i,
                share_with_layers,
                **(layer_kwargs or {}),
            )
            for i in range(n_interactions)
        ]
        self.n_up, self.n_down = n_up, n_down

    def init_state(self, shape, dtype):
        raise NotImplementedError

    def initial_embeddings(self):
        raise NotImplementedError

    def edge_feature_callback(
        self, edge_type, pos_sender, pos_receiver, sender_idx, receiver_idx
    ):
        raise NotImplementedError

    @classmethod
    @property
    def edge_types(cls):
        raise NotImplementedError

    def edge_factory(self, r, occupancies, n_occupancies):
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
        return edge_factory(r, occupancies, n_occupancies)

    @classmethod
    @property
    def layer_factory(cls):
        return MessagePassingLayer

    def __call__(self, r):
        occupancies = hk.get_state(
            'occupancies',
            shape=1,
            dtype=jnp.int32,
            init=self.init_state,
        )
        n_occupancies = hk.get_state(
            'n_occupancies', shape=[], dtype=jnp.int32, init=jnp.zeros
        )
        graph_edges, occupancies, n_occupancies = self.edge_factory(
            r, occupancies, n_occupancies
        )
        hk.set_state('occupancies', occupancies)
        hk.set_state('n_occupancies', n_occupancies)
        nuc_embedding, elec_embedding = self.initial_embeddings()
        graph = Graph(
            GraphNodes(nuc_embedding, elec_embedding),
            graph_edges,
        )

        for layer in self.layers:
            graph = layer(graph)

        return graph.nodes.electrons
