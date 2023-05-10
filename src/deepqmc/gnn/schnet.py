import haiku as hk
import jax.numpy as jnp
from jax import ops

from ..hkext import MLP
from ..utils import flatten
from .gnn import GraphNeuralNetwork, MessagePassingLayer
from .graph import GraphNodes, difference_callback


class SchNetLayer(MessagePassingLayer):
    r"""
    The message passing layer of :class:`ElectronGNN`.

    Derived from :class:`~deepqmc.gnn.gnn.MessagePassingLayer`.

    Args:
        residual (bool): whether a residual connection is used when updating
            the electron embeddings.
        convolution (bool): if :data:`True` the messages are generated via graph
            covolutions, else messages are generated from edge featues only.
        deep_features (bool): if :data:`true` edge features are updated through
            an MLP (:data:`u`), else initial edge features are reused.
        update_features (list[str]): which features to collect for the update
            of the electron embeddings.
            Possible values:

            - ``'residual'``: electron embedding from the previous interaction layer
            - ``'ne'``: sum over messages from nuclei
            - ``'same'``: sum over messages from electrons with same spin
            - ``'anti'``: sum over messages from electrons with opposite spin
            - ``'ee'``: sum of same and anti messeages
            - ``'nodes_up'``: sum over embeddings from spin-up electrons
            - ``'nodes_down'``: sum over embeddings from spin-down electrons

        update_rule (str): how to combine features for the update of the
            electron embeddings.
            Possible values:

            - ``'concatenate'``: run concatenated features through MLP
            - ``'featurewise'``: apply different MLP to each feature channel and sum
            - ``'featurewise_shared'``: apply the same MLP across feature channels
            - ``'sum'``: sum features before sending through an MLP
            note that `sum` and `featurewise_shared` imply features of same size

        subnet_kwargs (dict): extra arguments passed to the
            :class:`~deepqmc.hkext.MLP` constructor of the subnetworks.
        subnet_kwargs_by_lbl (dict): optional, extra arguments passed to the
            :class:`~deepqmc.hkext.MLP` constructor of the subnetworks. Arguments
            can be specified independently for each subnet
            (:data:`w`, :data:`h`, :data:`g` or :data:`u`).
    """

    def __init__(
        self,
        *,
        residual,
        convolution,
        deep_features,
        update_features,
        update_rule,
        subnet_kwargs=None,
        subnet_kwargs_by_lbl=None,
        **layer_attrs,
    ):
        super().__init__(**layer_attrs)
        assert update_rule in [
            'concatenate',
            'featurewise',
            'featurewise_shared',
            'sum',
        ]
        self.deep_features = deep_features
        self.update_features = update_features
        self.update_rule = update_rule
        self.convolution = convolution
        subnet_kwargs = subnet_kwargs or {}
        subnet_kwargs_by_lbl = subnet_kwargs_by_lbl or {}
        for lbl in self.subnet_labels:
            subnet_kwargs_by_lbl.setdefault(lbl, {})
            for k, v in subnet_kwargs.items():
                subnet_kwargs_by_lbl[lbl].setdefault(k, v)
            subnet_kwargs_by_lbl[lbl].setdefault('bias', lbl != 'w')
        if deep_features:
            self.u = {
                typ: MLP(
                    (
                        self.edge_feat_dim[typ]
                        if self.first_layer
                        else self.embedding_dim
                    ),
                    self.embedding_dim,
                    residual=not self.first_layer,
                    name=f'u{typ}',
                    **subnet_kwargs_by_lbl['u'],
                )
                for typ in self.edge_types
            }
        if self.convolution:
            self.w = {
                typ: MLP(
                    (
                        self.edge_feat_dim[typ]
                        if not deep_features
                        else self.embedding_dim
                    ),
                    self.embedding_dim,
                    name=f'w_{typ}',
                    **subnet_kwargs_by_lbl['w'],
                )
                for typ in self.edge_types
            }
            self.h = {
                typ: MLP(
                    self.embedding_dim,
                    self.embedding_dim,
                    name=f'h_{typ}',
                    **subnet_kwargs_by_lbl['h'],
                )
                for typ in self.edge_types
            }
        self.g = (
            MLP(
                self.embedding_dim
                * (len(self.update_features) if update_rule == 'concatenate' else 1),
                self.embedding_dim,
                name='g',
                **subnet_kwargs_by_lbl['g'],
            )
            if not self.update_rule == 'featurewise'
            else {
                uf: MLP(
                    self.embedding_dim,
                    self.embedding_dim,
                    name=f'g_{uf}',
                    **subnet_kwargs_by_lbl['g'],
                )
                for uf in (self.update_features)
            }
        )
        self.residual = residual

    @classmethod
    @property
    def subnet_labels(cls):
        return ('w', 'h', 'g', 'u')

    def get_update_edges_fn(self):
        def update_edges(edges):
            if self.deep_features:
                updated_edges = {
                    typ: edge._replace(features=self.u[typ](edge.features))
                    for typ, edge in edges.items()
                }
                return updated_edges
            else:
                return edges

        return update_edges

    def get_aggregate_edges_for_nodes_fn(self):
        def aggregate_edges_for_nodes(nodes, edges):
            if self.convolution:
                we = {typ: self.w[typ](edge.features) for typ, edge in edges.items()}
                hx = {
                    typ: (self.h[typ](self.mapping.sender_data_of(typ, nodes)))[
                        edges[typ].senders
                    ]
                    for typ in self.edge_types
                }
                message = {typ: we[typ] * hx[typ] for typ in self.edge_types}
            else:
                message = {typ: edge.features for typ, edge in edges.items()}

            z = {
                typ: ops.segment_sum(
                    data=message[typ],
                    segment_ids=edges[typ].receivers,
                    num_segments=self.mapping.receiver_data_of(typ, 'n_nodes'),
                )
                for typ in self.edge_types
            }
            return z

        return aggregate_edges_for_nodes

    def get_update_nodes_fn(self):
        def update_nodes(nodes, z):
            FEATURE_MAPPING = {
                'residual': nodes.electrons,
                'nodes_up': (
                    nodes.electrons[: self.n_up]
                    .mean(axis=0, keepdims=True)
                    .repeat(self.n_up + self.n_down, axis=0)
                ),
                'nodes_down': (
                    nodes.electrons[self.n_up :]
                    .mean(axis=0, keepdims=True)
                    .repeat(self.n_up + self.n_down, axis=0)
                ),
                'same': z['same'],
                'anti': z['anti'],
                'ee': z['same'] + z['anti'],
                'ne': z['ne'],
            }
            f = {uf: FEATURE_MAPPING[uf] for uf in self.update_features}
            if self.update_rule == 'concatenate':
                updated = self.g(jnp.concatenate(jnp.stack(list(f.values())), axis=-1))
            elif self.update_rule == 'featurewise':
                updated = jnp.sum(
                    jnp.stack([self.g[uf](f[uf]) for uf in self.update_features]),
                    axis=0,
                )
            elif self.update_rule == 'sum':
                updated = self.g(sum(f.values()))
            elif self.update_rule == 'featurewise_shared':
                updated = jnp.sum(self.g(jnp.stack(list(f.values()))), axis=0)
            if self.residual:
                updated = updated + nodes.electrons
            nodes = GraphNodes(nodes.nuclei, updated)

            return nodes

        return update_nodes


class SchNet(GraphNeuralNetwork):
    r"""
    The SchNet architecture [SchuettJCP18]_ adapted for graphs of nuclei and electrons.

    Derived from :class:`~deepqmc.gnn.gnn.GraphNeuralNetwork`.

    Args:
        mol (~deepqmc.Molecule): the molecule on which the graph is defined.
        embedding_dim (int): the length of the electron embedding vectors.
        n_interactions (int): number of message passing interactions.
        posisional_electron_embeddings(bool): whether to initialize the electron
            embbedings with the concatenated edge features.
        edge_feat_factory:
        gnn_kwargs (dict): extra arguments passed to the
            :class:`~deepqmc.gnn.gnn.GraphNeuralNetwork` base class.
    """

    def __init__(
        self,
        mol,
        embedding_dim,
        *,
        n_interactions,
        positional_electron_embeddings,
        edge_feat_factory,
        **gnn_kwargs,
    ):
        n_nuc, n_up, n_down = mol.n_particles
        super().__init__(
            mol,
            embedding_dim,
            n_interactions,
            **gnn_kwargs,
        )
        self.edge_features = {
            typ: (
                edge_feat_factory[typ]()
                if isinstance(edge_feat_factory, dict)
                else edge_feat_factory()
            )
            for typ in self.edge_types
        }
        self.positional_electron_embeddings = positional_electron_embeddings

    def node_factory(self, edges):
        n_elec_types = self.node_data['n_node_types']['electrons']
        n_nuc_types = self.node_data['n_node_types']['nuclei']
        if self.positional_electron_embeddings:
            X = hk.Linear(
                output_size=self.embedding_dim,
                with_bias=False,
                name='ElectronicEmbedding',
            )
            ne_pos_feat = (
                jnp.zeros(
                    (
                        self.n_up + self.n_down + 1,
                        self.n_nuc + 1,
                        edges['ne'].features.shape[-1],
                    )
                )
                .at[edges['ne'].receivers, edges['ne'].senders]
                .set(edges['ne'].features)[: self.n_up + self.n_down, : self.n_nuc]
            )  # [n_elec, n_nuc, n_edge_feat_dim]
            x = X(flatten(ne_pos_feat, start_axis=1))
        else:
            X = hk.Embed(n_elec_types, self.embedding_dim, name='ElectronicEmbedding')
            x = X(self.node_data['node_types']['electrons'])
        Y = hk.Embed(n_nuc_types, self.embedding_dim, name='NuclearEmbedding')
        y = Y(self.node_data['node_types']['nuclei'] - n_elec_types)
        return GraphNodes(y, x)

    @classmethod
    @property
    def edge_types(cls):
        return ('same', 'anti', 'ne')

    def edge_feature_callback(self, typ, *feature_callback_args):
        r = difference_callback(*feature_callback_args)
        return self.edge_features[typ](r)

    @classmethod
    @property
    def layer_factory(cls):
        return SchNetLayer
