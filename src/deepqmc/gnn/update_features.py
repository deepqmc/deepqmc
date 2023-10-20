from typing import Sequence

import haiku as hk
import jax.numpy as jnp

from ..hkext import Identity


class UpdateFeature(hk.Module):
    r"""Base class for all update features."""

    def __init__(self, n_up, n_down, two_particle_stream_dim, node_edge_mapping):
        super().__init__()
        self.n_up = n_up
        self.n_down = n_down
        self.node_edge_mapping = node_edge_mapping
        self.two_particle_stream_dim = two_particle_stream_dim

    @property
    def names(self) -> Sequence[str]:
        raise NotImplementedError

    def __call__(self, nodes, edges) -> Sequence[jnp.ndarray]:
        raise NotImplementedError


class ResidualUpdateFeature(UpdateFeature):
    r"""Residual update feature.

    Returns the unchanged electron embeddings from the previous layer as
    a single update feature.
    """

    def __call__(self, nodes, edges) -> Sequence[jnp.ndarray]:
        return [nodes.electrons]

    @property
    def names(self):
        return ['residual']


class NodeSumUpdateFeature(UpdateFeature):
    r"""The (normalized) sum of the node embeddings as an update feature.

    Returns the (normalized) sum of the electron embeddings from the previous layer as
    a single update feature.
    """

    def __init__(self, *args, node_types, normalize):
        assert all(node_type in {'up', 'down'} for node_type in node_types)
        super().__init__(*args)
        self.normalize = normalize
        self.node_types = node_types

    def __call__(self, nodes, edges) -> Sequence[jnp.ndarray]:
        node_idx = {'up': slice(None, self.n_up), 'down': slice(self.n_up, None)}
        reduce_fn = jnp.mean if self.normalize else jnp.sum
        return [
            jnp.tile(
                reduce_fn(nodes.electrons[node_idx[node_type]], axis=0, keepdims=True),
                (self.n_up + self.n_down, 1),
            )
            for node_type in self.node_types
        ]

    @property
    def names(self):
        return [f'node_{node_type}' for node_type in self.node_types]


class EdgeSumUpdateFeature(UpdateFeature):
    r"""The (normalized) sum of the edge embeddings as an update feature.

    Returns the (normalized) sum of the edge embeddings for various edge types
    as separate update features.
    """

    def __init__(self, *args, edge_types, normalize):
        assert all(
            edge_type in {'up', 'down', 'same', 'anti', 'ee', 'ne'}
            for edge_type in edge_types
        )
        super().__init__(*args)
        self.normalize = normalize
        self.edge_types = edge_types

    def __call__(self, nodes, edges) -> Sequence[jnp.ndarray]:
        updates = []
        for edge_type in self.edge_types:
            if edge_type == 'ee':
                factor = self.n_up + self.n_down if self.normalize else 1.0
                updates.append(
                    (
                        edges['same'].sum_senders(False)
                        + edges['anti'].sum_senders(False)
                    )
                    / factor
                )
            else:
                updates.append(edges[edge_type].sum_senders(self.normalize))
        return updates

    @property
    def names(self):
        return [f'edge_{edge_type}' for edge_type in self.edge_types]


class ConvolutionUpdateFeature(UpdateFeature):
    r"""The convolution of node and edge embeddings as an update feature.

    Returns the convolution of the node and edge embeddings for various edge types
    as separate update features.
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

    def __call__(self, nodes, edges) -> Sequence[jnp.ndarray]:
        updates = []
        for edge_type in self.edge_types:
            if edge_type == 'ee':
                ee = sum(
                    self.single_edge_type_update(nodes, edges, st, False)
                    for st in ['same', 'anti']
                )
                factor = self.n_up + self.n_down if self.normalize else 1.0
                updates.append(ee / factor)
            else:
                updates.append(
                    self.single_edge_type_update(
                        nodes, edges, edge_type, self.normalize
                    )
                )
        return updates

    @property
    def names(self):
        return [f'conv_{edge_type}' for edge_type in self.edge_types]


class NodeAttentionUpdateFeature(UpdateFeature):
    r"""Create a single update feature by attenting over the nodes.

    Returns the Psiformer update feature based on attention over the nodes.
    """

    def __init__(self, *args, num_heads, mlp_factory, attention_residual, mlp_residual):
        super().__init__(*args)
        self.num_heads = num_heads
        self.attention_residual = attention_residual
        self.mlp_residual = mlp_residual
        self.mlp_factory = mlp_factory

    def __call__(self, nodes, edges):
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
        return [mlp_out]
