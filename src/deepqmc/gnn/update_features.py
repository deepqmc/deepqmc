from collections.abc import Mapping, Sequence

import haiku as hk
import jax.numpy as jnp

from ..hkext import Identity
from .graph import GraphEdges, GraphNodes
from .utils import NodeEdgeMapping


class UpdateFeature(hk.Module):
    r"""Base class for all update features.

    Args:
        n_up (int): number of spin up electrons
        n_down (int): number of spin down electrons
        two_particle_stream_dim (int): dimension of the two-particle stream
        node_edge_mapping (~deepqmc.gnn.utils.NodeEdgeMapping): mapping between the
            various node and edge types.
    """

    def __init__(
        self,
        n_up: int,
        n_down: int,
        two_particle_stream_dim: int,
        node_edge_mapping: NodeEdgeMapping,
    ):
        super().__init__()
        self.n_up = n_up
        self.n_down = n_down
        self.node_edge_mapping = node_edge_mapping
        self.two_particle_stream_dim = two_particle_stream_dim

    @property
    def names(self) -> list[str]:
        raise NotImplementedError

    def __call__(
        self, nodes: GraphNodes, edges: Mapping[str, GraphEdges]
    ) -> Sequence[GraphNodes]:
        raise NotImplementedError


class ResidualElectronUpdateFeature(UpdateFeature):
    r"""Residual update feature.

    Returns the unchanged electron embeddings from the previous layer as
    a single update feature.
    """

    def __call__(
        self, nodes: GraphNodes, edges: Mapping[str, GraphEdges]
    ) -> Sequence[GraphNodes]:
        return [GraphNodes(None, nodes.electrons)]

    @property
    def names(self) -> list[str]:
        return ['residual']


class NodeSumElectronUpdateFeature(UpdateFeature):
    r"""The (normalized) sum of the node embeddings as an update feature.

    Returns the (normalized) sum of the electron embeddings from the previous layer as
    a single update feature.

    Args:
        n_up (int): number of spin up electrons
        n_down (int): number of spin down electrons
        two_particle_stream_dim (int): dimension of the two-particle stream
        node_edge_mapping (~deepqmc.gnn.utils.NodeEdgeMapping): mapping between the
            various node and edge types.
        node_types (list[str]): list of node types to update
        normalize (bool): whether to normalize the sum by the number of nodes
    """

    def __init__(self, *args, node_types, normalize):
        assert all(node_type in {'up', 'down'} for node_type in node_types)
        super().__init__(*args)
        self.normalize = normalize
        self.node_types = node_types

    def __call__(
        self, nodes: GraphNodes, edges: Mapping[str, GraphEdges]
    ) -> Sequence[GraphNodes]:
        node_idx = {'up': slice(None, self.n_up), 'down': slice(self.n_up, None)}
        reduce_fn = jnp.mean if self.normalize else jnp.sum
        return [
            GraphNodes(
                None,
                jnp.tile(
                    reduce_fn(
                        nodes.electrons[node_idx[node_type]], axis=0, keepdims=True
                    ),
                    (self.n_up + self.n_down, 1),
                ),
            )
            for node_type in self.node_types
        ]

    @property
    def names(self) -> list[str]:
        return [f'node_{node_type}' for node_type in self.node_types]


class EdgeSumElectronUpdateFeature(UpdateFeature):
    r"""The (normalized) sum of the edge embeddings as an update feature.

    Returns the (normalized) sum of the edge embeddings for various edge types
    as separate update features.

    Args:
        n_up (int): number of spin up electrons
        n_down (int): number of spin down electrons
        two_particle_stream_dim (int): dimension of the two-particle stream
        node_edge_mapping (~deepqmc.gnn.utils.NodeEdgeMapping): mapping between the
            various node and edge types.
        edge_types (list[str]): list of edge types to sum over
        normalize (bool): whether to normalize the sum by the number of senders
    """

    def __init__(self, *args, edge_types, normalize):
        assert all(
            edge_type in {'up', 'down', 'same', 'anti', 'ee', 'ne'}
            for edge_type in edge_types
        )
        super().__init__(*args)
        self.normalize = normalize
        self.edge_types = edge_types

    def __call__(
        self, nodes: GraphNodes, edges: Mapping[str, GraphEdges]
    ) -> Sequence[GraphNodes]:
        updates = []
        for edge_type in self.edge_types:
            if edge_type == 'ee':
                factor = self.n_up + self.n_down if self.normalize else 1.0
                updates.append(
                    GraphNodes(
                        None,
                        (
                            edges['same'].sum_senders(False)
                            + edges['anti'].sum_senders(False)
                        )
                        / factor,
                    )
                )
            else:
                updates.append(
                    GraphNodes(None, edges[edge_type].sum_senders(self.normalize))
                )
        return updates

    @property
    def names(self) -> list[str]:
        return [f'edge_{edge_type}' for edge_type in self.edge_types]


class ConvolutionElectronUpdateFeature(UpdateFeature):
    r"""The convolution of node and edge embeddings as an update feature.

    Returns the convolution of the node and edge embeddings for various edge types
    as separate update features.

    Args:
        n_up (int): number of spin up electrons
        n_down (int): number of spin down electrons
        two_particle_stream_dim (int): dimension of the two-particle stream
        node_edge_mapping (~deepqmc.gnn.utils.NodeEdgeMapping): mapping between the
            various node and edge types.
        edge_types (list[str]): list of edge types to sum over
        normalize (bool): whether to normalize the sum by the number of senders
        w_factory (~collections.abc.Callable): factory function for the :math:`w` matrix
        h_factory (~collections.abc.Callable): factory function for the :math:`h` matrix
        w_for_ne (bool): whether to use the :math:`w` matrix for the :math:`ne` edge
            type
    """

    def __init__(
        self, *args, edge_types, normalize, w_factory, h_factory, w_for_ne=True
    ):
        assert all(
            edge_type in {'up', 'down', 'same', 'anti', 'ee', 'ne'}
            for edge_type in edge_types
        )
        super().__init__(*args)
        self.normalize = normalize
        self.edge_types = edge_types
        layer_types = [typ for typ in edge_types if typ != 'ee']
        if 'ee' in edge_types:
            layer_types.extend(['same', 'anti'])
        self.h_factory = h_factory
        self.w_factory = w_factory
        self.w_for_ne = w_for_ne

    def single_edge_type_update(self, nodes, edges, edge_type, normalize):
        w = (
            self.w_factory(self.two_particle_stream_dim, name=f'w_{edge_type}')
            if self.w_for_ne or edge_type != 'ne'
            else Identity()
        )
        we = w(edges[edge_type].single_array)
        h = self.h_factory(we.shape[-1], name=f'h_{edge_type}')
        hx = h(self.node_edge_mapping.sender_data_of(edge_type, nodes))
        if edges[edge_type].single_array.size == 0:
            # parameters acting on size zero arrays cause NaN gradients
            return jnp.zeros((hx.shape[0], self.two_particle_stream_dim))
        return edges[edge_type].update_from_single_array(we).convolve(hx, normalize)

    def __call__(
        self, nodes: GraphNodes, edges: Mapping[str, GraphEdges]
    ) -> Sequence[GraphNodes]:
        updates = []
        for edge_type in self.edge_types:
            if edge_type == 'ee':
                ee = sum(
                    self.single_edge_type_update(nodes, edges, st, False)
                    for st in ['same', 'anti']
                )
                factor = self.n_up + self.n_down if self.normalize else 1.0
                updates.append(GraphNodes(None, ee / factor))
            else:
                updates.append(
                    GraphNodes(
                        None,
                        self.single_edge_type_update(
                            nodes, edges, edge_type, self.normalize
                        ),
                    )
                )
        return updates

    @property
    def names(self) -> list[str]:
        return [f'conv_{edge_type}' for edge_type in self.edge_types]


class NodeAttentionElectronUpdateFeature(UpdateFeature):
    r"""Create a single update feature by attenting over the nodes.

    Returns the Psiformer update feature based on attention over the nodes.

    Args:
        n_up (int): number of spin up electrons
        n_down (int): number of spin down electrons
        two_particle_stream_dim (int): dimension of the two-particle stream
        node_edge_mapping (~deepqmc.gnn.utils.NodeEdgeMapping): mapping between the
            various node and edge types.
        num_heads (int): number of attention heads
        mlp_factory (~typing.Type[~deepqmc.hkext.MLP]): factory function for the MLP
        attention_residual (Optional[~deepqmc.hkext.Residual]): optional residual
            connection after the attention layer
        mlp_residual (Optional[~deepqmc.hkext.Residual]): optional residual
            connection after the MLP layer
    """

    def __init__(self, *args, num_heads, mlp_factory, attention_residual, mlp_residual):
        super().__init__(*args)
        self.num_heads = num_heads
        self.attention_residual = attention_residual
        self.mlp_residual = mlp_residual
        self.mlp_factory = mlp_factory

    def __call__(
        self, nodes: GraphNodes, edges: Mapping[str, GraphEdges]
    ) -> Sequence[GraphNodes]:
        h = nodes.electrons
        heads_dim = h.shape[-1] // self.num_heads
        assert heads_dim * self.num_heads == h.shape[-1]
        attention_layer = hk.MultiHeadAttention(
            self.num_heads,
            heads_dim,
            w_init=hk.initializers.VarianceScaling(1, 'fan_in', 'normal'),
            with_bias=False,
        )
        mlp = self.mlp_factory(h.shape[-1], name='mlp')
        attended = attention_layer(h, h, h)
        if self.attention_residual:
            attended = self.attention_residual(h, attended)
        mlp_out = mlp(attended)
        if self.mlp_residual:
            mlp_out = self.mlp_residual(attended, mlp_out)
        return [GraphNodes(None, mlp_out)]


class CombinedNodeAttentionUpdateFeature(UpdateFeature):
    r"""Create an attention update feature for both electrons and nuclei.

    The update feature is created by attending over both electrons and nuclei.

    Args:
        n_up (int): number of spin up electrons
        n_down (int): number of spin down electrons
        two_particle_stream_dim (int): dimension of the two-particle stream
        node_edge_mapping (~deepqmc.gnn.utils.NodeEdgeMapping): mapping between the
            various node and edge types.
        num_heads (int): number of attention heads
        mlp_factory (~typing.Type[~deepqmc.hkext.MLP]): factory function for the MLP
        attention_residual (~deepqmc.hkext.Residual): optional, optional residual
            connection after the attention layer
        mlp_residual (~deepqmc.hkext.Residual): optional, residual connection after the
            MLP layer
        elec_to_nuc (bool): whether to allow attention over the electrons
            to influence the nuclei
    """

    def __init__(
        self,
        *args,
        num_heads,
        mlp_factory,
        attention_residual,
        mlp_residual,
        elec_to_nuc,
    ):
        super().__init__(*args)
        self.num_heads = num_heads
        self.attention_residual = attention_residual
        self.mlp_residual = mlp_residual
        self.mlp_factory = mlp_factory
        self.elec_to_nuc = elec_to_nuc

    def __call__(
        self, nodes: GraphNodes, edges: Mapping[str, GraphEdges]
    ) -> Sequence[GraphNodes]:
        n_nuc = len(nodes.nuclei)
        n_el = len(nodes.electrons)
        h = jnp.concatenate([nodes.nuclei, nodes.electrons], axis=0)
        mask = (
            None
            if self.elec_to_nuc
            else jnp.ones((1, n_nuc + n_el, n_nuc + n_el), dtype=bool)
            .at[:, :n_nuc, n_nuc:]
            .set(False)
        )
        heads_dim = h.shape[-1] // self.num_heads
        assert heads_dim * self.num_heads == h.shape[-1]
        attention_layer = hk.MultiHeadAttention(
            self.num_heads,
            heads_dim,
            w_init=hk.initializers.VarianceScaling(1, 'fan_in', 'normal'),
            with_bias=False,
        )
        mlp = self.mlp_factory(h.shape[-1], name='mlp')
        attended = attention_layer(h, h, h, mask)
        if self.attention_residual:
            attended = self.attention_residual(h, attended)
        mlp_out = mlp(attended)
        if self.mlp_residual:
            mlp_out = self.mlp_residual(attended, mlp_out)
        nuclei_out, electrons_out = jnp.split(mlp_out, [nodes.nuclei.shape[0]], axis=0)
        return [GraphNodes(nuclei_out, electrons_out)]
