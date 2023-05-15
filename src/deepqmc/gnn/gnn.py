import haiku as hk

from .graph import GraphUpdate
from .utils import NodeEdgeMapping


class MessagePassingLayer(hk.Module):
    r"""
    Base class for all message passing layers.

    Args:
        ilayer (int): the index of the current layer in the list of all layers
        shared (dict): attribute names and values which are shared between the
            layers and the :class:`GraphNeuralNetwork` instance.
    """

    def __init__(
        self,
        ilayer,
        n_nuc,
        n_up,
        n_down,
        embedding_dim,
        n_interactions,
        edge_types,
        node_data,
        **layer_attrs,
    ):
        super().__init__()
        self.n_nuc, self.n_up, self.n_down = n_nuc, n_up, n_down
        self.embedding_dim = embedding_dim
        self.first_layer = ilayer == 0
        last_layer = ilayer == n_interactions - 1
        self.edge_types = tuple(
            typ for typ in edge_types if not last_layer or typ not in {'nn', 'en'}
        )
        for name, attr in layer_attrs.items():
            setattr(self, name, attr)
        self.mapping = NodeEdgeMapping(self.edge_types, node_data=node_data)
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
