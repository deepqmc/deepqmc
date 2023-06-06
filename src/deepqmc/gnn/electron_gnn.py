from functools import partial
from itertools import accumulate

import haiku as hk
import jax.numpy as jnp
from jax import tree_util

from ..hkext import MLP, Identity
from ..utils import flatten
from .graph import (
    Graph,
    GraphNodes,
    GraphUpdate,
    MolecularGraphEdgeBuilder,
    offdiagonal_sender_idx,
)
from .utils import NodeEdgeMapping


class ElectronGNNLayer(hk.Module):
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
            - ``'edge_ne'``: sum over messages from nuclei
            - ``'edge_same'``: sum over messages from electrons with same spin
            - ``'edge_anti'``: sum over messages from electrons with opposite spin
            - ``'edge_up'``: sum over messages from spin up electrons
            - ``'edge_down'``: sum over messages from spin down electrons
            - ``'edge_ee'``: sum of over messages from all electrons
            - ``'node_up'``: sum over embeddings from spin-up electrons
            - ``'node_down'``: sum over embeddings from spin-down electrons

        update_rule (str): how to combine features for the update of the
            electron embeddings.
            Possible values:

            - ``'concatenate'``: run concatenated features through MLP
            - ``'featurewise'``: apply different MLP to each feature channel and sum
            - ``'featurewise_shared'``: apply the same MLP across feature channels
            - ``'sum'``: sum features before sending through an MLP

            note that `sum` and `featurewise_shared` imply features of same size

        subnet_factory (Callable): A function that constructs the subnetworks of
            the GNN layer.
        subnet_factory_by_lbl (dict): optional, a dictionary of functions that construct
            subnetworks of the GNN layer. If both this and :data:`subnet_factory` is
            specified, the specified values of :data:`subnet_factory_by_lbl` will take
            precedence. If some keys are missing, the default value of
            :data:`subnet_factory` will be used in their place. Possible keys are:
            (:data:`w`, :data:`h`, :data:`g` or :data:`u`).
    """

    def __init__(
        self,
        n_interactions,
        ilayer,
        n_nuc,
        n_up,
        n_down,
        embedding_dim,
        edge_types,
        self_interaction,
        node_data,
        edge_feat_dim,
        two_particle_stream_dim,
        *,
        one_particle_residual,
        two_particle_residual,
        convolution,
        deep_features,
        mean_aggregate_edges,
        update_features,
        update_rule,
        w_for_ne=True,
        subnet_factory=None,
        subnet_factory_by_lbl=None,
    ):
        super().__init__()
        self.n_nuc, self.n_up, self.n_down = n_nuc, n_up, n_down
        self.last_layer = ilayer == n_interactions - 1
        self.edge_types = tuple(
            typ for typ in edge_types if not self.last_layer or typ not in {'nn', 'en'}
        )
        self.mapping = NodeEdgeMapping(self.edge_types, node_data=node_data)
        assert update_rule in [
            'concatenate',
            'featurewise',
            'featurewise_shared',
            'sum',
        ]
        assert all(
            uf
            in [
                'residual',
                'edge_ne',
                'edge_same',
                'edge_anti',
                'edge_up',
                'edge_down',
                'edge_ee',
                'node_up',
                'node_down',
                'convolution_same',
                'convolution_anti',
                'convolution_ne',
            ]
            for uf in update_features
        )
        assert (
            update_rule not in ['sum', 'featurewise_shared']
            or embedding_dim == two_particle_stream_dim
        )
        assert deep_features in [False, 'shared', 'separate']
        self.deep_features = deep_features
        self.update_features = update_features
        self.update_rule = update_rule
        self.convolution = convolution
        subnet_factory_by_lbl = subnet_factory_by_lbl or {}
        for lbl in ['w', 'h', 'g', 'u']:
            subnet_factory_by_lbl.setdefault(lbl, subnet_factory)
        if deep_features:
            self.u = (
                subnet_factory_by_lbl['u'](two_particle_stream_dim, name='u')
                if deep_features == 'shared'
                else {
                    typ: subnet_factory_by_lbl['u'](
                        two_particle_stream_dim,
                        name=f'u{typ}',
                    )
                    for typ in self.edge_types
                }
            )
        if self.convolution:
            self.w = {
                typ: (
                    subnet_factory_by_lbl['w'](
                        two_particle_stream_dim,
                        name=f'w_{typ}',
                    )
                    if w_for_ne or typ != 'ne'
                    else Identity()
                )
                for typ in self.edge_types
            }
            self.h = {
                typ: subnet_factory_by_lbl['h'](
                    (
                        edge_feat_dim['ne']
                        if not w_for_ne and typ == 'ne' and ilayer == 0
                        else two_particle_stream_dim
                    ),
                    name=f'h_{typ}',
                )
                for typ in self.edge_types
            }
        self.g = (
            subnet_factory_by_lbl['g'](
                embedding_dim,
                name='g',
            )
            if not self.update_rule == 'featurewise'
            else {
                uf: subnet_factory_by_lbl['g'](
                    embedding_dim,
                    name=f'g_{uf}',
                )
                for uf in (self.update_features)
            }
        )
        self.one_particle_residual = one_particle_residual
        self.two_particle_residual = two_particle_residual
        self.self_interaction = self_interaction
        self.mean_aggregate_edges = mean_aggregate_edges

    def get_update_edges_fn(self):
        def update_edges(edges):
            if self.deep_features:
                if self.deep_features == 'shared':
                    # combine features along leading dim, apply MLP and split
                    # into channels again to please kfac
                    keys, feats = zip(*edges.items())
                    split_idxs = list(accumulate(len(f) for f in feats))
                    feats = jnp.split(self.u(jnp.concatenate(feats)), split_idxs)
                    updated_edges = dict(zip(keys, feats))
                elif self.deep_features == 'separate':
                    updated_edges = {
                        typ: self.u[typ](edge) for typ, edge in edges.items()
                    }

                if self.two_particle_residual:
                    updated_edges = self.two_particle_residual(
                        edges, updated_edges
                    )
                return updated_edges
            else:
                return edges

        return update_edges

    def get_aggregate_edges_for_nodes_fn(self):
        def aggregate_edges_for_nodes(nodes, edges):
            reduce_fn = jnp.mean if self.mean_aggregate_edges else jnp.sum
            def edge_update_feature(edge_type):
                return edges[edge_type].mean(axis=0)

            def convolve_ne(w, h):
                return reduce_fn(w * h[:, None], axis=0)

            def convolve_same(w, h):
                #  w_uu, w_dd = jnp.split(w, (self.n_up**2,))
                w_uu, w_dd = jnp.split(w, (self.n_up * (self.n_up - 1),))
                wh = jnp.concatenate(
                    [
                        reduce_fn(
                            w_uu.reshape(self.n_up - 1, self.n_up, -1)
                            #  * h[: self.n_up, None]
                            * h[offdiagonal_sender_idx(self.n_up)],
                            axis=0
                        ),
                        reduce_fn(
                            w_dd.reshape(self.n_down - 1, self.n_down, -1)
                            #  * h[self.n_up :, None]
                            * h[self.n_up + offdiagonal_sender_idx(self.n_down)],
                            axis=0
                        )
                    ]
                )
                return wh

            def convolve_anti(w, h):
                w_ud, w_du = jnp.split(w, 2)
                wh = jnp.concatenate(
                    [
                        reduce_fn(
                            w_du.reshape(self.n_down, self.n_up, -1)
                            * h[self.n_up :, None],
                            axis=0
                        ),
                       reduce_fn(
                            w_ud.reshape(self.n_up, self.n_down, -1)
                            * h[: self.n_up, None],
                            axis=0
                        )
                    ]
                )
                return wh

            def convolution(edge_type):
                hx = self.h[edge_type](self.mapping.sender_data_of(edge_type, nodes))
                we = self.w[edge_type](edges[edge_type])
                return {
                    'same': convolve_same,
                    'anti': convolve_anti,
                    'ne': convolve_ne,
                }[edge_type](we, hx)

            UPDATE_FEATURES = {
                'residual': lambda: nodes.electrons,
                'node_up': lambda: (
                    nodes.electrons[: self.n_up]
                    .mean(axis=0, keepdims=True)
                    .repeat(self.n_up + self.n_down, axis=0)
                ),
                'node_down': lambda: (
                    nodes.electrons[self.n_up :]
                    .mean(axis=0, keepdims=True)
                    .repeat(self.n_up + self.n_down, axis=0)
                ),
                'edge_same': partial(edge_update_feature, 'same'),
                'edge_anti': partial(edge_update_feature, 'anti'),
                'edge_up': partial(edge_update_feature, 'up'),
                'edge_down': partial(edge_update_feature, 'down'),
                'edge_ne': partial(edge_update_feature, 'ne'),
                'edge_ee': lambda: 0.5 * (
                    edge_update_feature('same') + edge_update_feature('anti')
                ),
                'convolution_same': partial(convolution, 'same'),
                'convolution_anti': partial(convolution, 'anti'),
                'convolution_ne': partial(convolution, 'ne'),
                'convolution_ee': lambda: 0.5 * (
                    convolution('same') + convolution('anti')
                ),
            }

            return {uf: UPDATE_FEATURES[uf]() for uf in self.update_features}

        return aggregate_edges_for_nodes

    def get_update_nodes_fn(self):
        def update_nodes(nodes, f):
            if self.update_rule == 'concatenate':
                updated = self.g(
                    jnp.concatenate([f[uf] for uf in self.update_features], axis=-1)
                )
            elif self.update_rule == 'featurewise':
                updated = sum(self.g[uf](f[uf]) for uf in self.update_features)
            elif self.update_rule == 'sum':
                updated = self.g(sum(f.values()))
            elif self.update_rule == 'featurewise_shared':
                updated = jnp.sum(self.g(jnp.stack(list(f.values()))), axis=0)
            if self.one_particle_residual:
                updated = self.one_particle_residual(nodes.electrons, updated)
            nodes = GraphNodes(nodes.nuclei, updated)

            return nodes

        return update_nodes

    def __call__(self, graph):
        r"""
        Execute the message passing layer.

        Args:
            graph (:class:`Graph`)

        Returns:
            :class:`Graph`: updated graph
        """
        update_graph = GraphUpdate(
            update_nodes_fn=self.get_update_nodes_fn(),
            update_edges_fn=None if self.last_layer else self.get_update_edges_fn(),
            aggregate_edges_for_nodes_fn=self.get_aggregate_edges_for_nodes_fn(),
        )
        return update_graph(graph)


class ElectronGNN(hk.Module):
    r"""
    A neural network acting on graphs defined by electrons and nuclei.

    Derived from :class:`~deepqmc.gnn.gnn.GraphNeuralNetwork`.

    Args:
        mol (:class:`~deepqmc.Molecule`): the molecule on which the graph is defined.
        embedding_dim (int): the length of the electron embedding vectors.
        n_interactions (int): number of message passing interactions.
        positional_electron_embeddings(bool): whether to initialize the electron
            embbedings with the concatenated edge features.
        edge_features: a function or a :data:`dict` of functions for each edge
            type, embedding the interparticle differences.
        edge_types: the types of edges to consider in the molecular graph. It should
            be a sequence of unique :data:`str`s from the follwing options:
            - ``'nn'``: nucleus-nucleus edges
            - ``'ne'``: nucleus-electron edges
            - ``'en'``: electron-nucleus edges
            - ``'same'``: electron-electron edges between electrons of the same spin
            - ``'anti'``: electron-electron edges between electrons of opposite spins
        two_particle_stream_dim (int): the feature dimension of the two particle
            streams. Only active if :data:`deep_features` are used.
        layer_factory (Callable): a callable that generates a layer of the GNN.
        ghost_coords: optional, specifies the coordinates of one or more ghost atoms,
            useful for breaking spatial symmetries of the nuclear geometry.
    """

    def __init__(
        self,
        mol,
        embedding_dim,
        *,
        n_interactions,
        positional_electron_embeddings,
        edge_features,
        edge_types,
        self_interaction,
        two_particle_stream_dim,
        nuclei_embedding,
        layer_factory,
        ghost_coords=None,
    ):
        super().__init__()
        n_nuc, n_up, n_down = mol.n_particles
        edge_feat_dim = {typ: len(edge_features[typ]) for typ in edge_types}
        n_atom_types = mol.n_atom_types
        charges = mol.charges
        self.ghost_coords = None
        if ghost_coords is not None:
            charges = jnp.concatenate([charges, jnp.zeros(len(ghost_coords))])
            n_nuc += len(ghost_coords)
            n_atom_types += 1
            self.ghost_coords = jnp.asarray(ghost_coords)
        self.n_nuc, self.n_up, self.n_down = n_nuc, n_up, n_down
        self.embedding_dim = embedding_dim
        self.node_data = {
            'n_nodes': {'nuclei': n_nuc, 'electrons': n_up + n_down},
            'n_node_types': {'electrons': 1 if n_up == n_down else 2},
            'node_types': {
                'electrons': jnp.array(n_up * [0] + n_down * [int(n_up != n_down)])
            },
        }
        self.layers = [
            layer_factory(
                n_interactions,
                ilayer,
                n_nuc,
                n_up,
                n_down,
                embedding_dim,
                edge_types,
                self_interaction,
                self.node_data,
                edge_feat_dim,
                two_particle_stream_dim,
            )
            for ilayer in range(n_interactions)
        ]
        self.edge_features = edge_features
        self.edge_types = edge_types
        self.positional_electron_embeddings = positional_electron_embeddings
        self.nuclei_embedding = (
            nuclei_embedding(charges, n_atom_types) if nuclei_embedding else None
        )
        self.self_interaction = self_interaction

    def node_factory(self, phys_conf):
        n_elec_types = self.node_data['n_node_types']['electrons']
        if self.positional_electron_embeddings:
            edge_factory = MolecularGraphEdgeBuilder(
                self.n_nuc, self.n_up, self.n_down, self.positional_electron_embeddings.keys(), self_interaction=self.self_interaction,
            )
            feats = tree_util.tree_map(
                lambda f, e: f(e).swapaxes(0, 1).reshape(self.n_up + self.n_down, -1),
                self.positional_electron_embeddings,
                edge_factory(phys_conf),
            )
            x = tree_util.tree_reduce(partial(jnp.concatenate, axis=1), feats)
        else:
            X = hk.Embed(n_elec_types, self.embedding_dim, name='ElectronicEmbedding')
            x = X(self.node_data['node_types']['electrons'])
        return (
            GraphNodes(self.nuclei_embedding(), x)
            if self.nuclei_embedding
            else GraphNodes(None, x)
        )

    def edge_factory(self, phys_conf):
        r"""Compute all the graph edges used in the GNN."""

        edge_factory = MolecularGraphEdgeBuilder(
            self.n_nuc,
            self.n_up,
            self.n_down,
            self.edge_types,
            self_interaction=self.self_interaction,
        )
        edges = edge_factory(phys_conf)
        return {typ: self.edge_features[typ](edges[typ]) for typ in self.edge_types}

    def __call__(self, phys_conf):
        r"""
        Execute the graph neural network.

        Args:
            phys_conf (PhysicalConfiguration): the physical configuration
                of the molecule.

        Returns:
            float, (:math:`N_\text{elec}`, :data:`embedding_dim`):
            the final embeddings of the electrons.
        """
        if self.ghost_coords is not None:
            phys_conf = phys_conf._replace(
                R=jnp.concatenate(
                    [
                        phys_conf.R,
                        jnp.tile(self.ghost_coords[None], (len(phys_conf.R), 1, 1)),
                    ],
                    axis=-2,
                )
            )
        graph_edges = self.edge_factory(phys_conf)
        graph_nodes = self.node_factory(phys_conf)
        graph = Graph(graph_nodes, graph_edges)

        for layer in self.layers:
            graph = layer(graph)

        return graph.nodes.electrons


class NucleiEmbedding(hk.Module):
    r"""Create initial embeddings for nuclei.

    Args:
        charges (jnp.ndarray): the nuclear charges of the molecule.
        n_atom_types (int): the number of different atom types in the molecule.
        embedding_dim (int): the length of the output embedding vector
        atom_type_embedding (bool): if :data:`True`, initial embeddings are the same
            for atoms of the same type (nuclear charge), otherwise they are different
            for all nuclei.
        subnet_type (str): the type of subnetwork to use for the embedding generation:
            - ``'mlp'``: an MLP is used
            - ``'embed'``: a :class:`haiku.Embed` block is used
    """

    def __init__(
        self, charges, n_atom_types, *, embedding_dim, atom_type_embedding, subnet_type
    ):
        super().__init__()
        assert subnet_type in ['mlp', 'embed']
        n_nuc_types = n_atom_types if atom_type_embedding else len(charges)
        if subnet_type == 'mlp':
            self.subnet = MLP(
                embedding_dim,
                hidden_layers=['log', 1],
                bias=True,
                last_linear=False,
                activation=jnp.tanh,
                init='deeperwin',
            )
        elif subnet_type == 'embed':
            self.subnet = hk.Embed(n_nuc_types, embedding_dim)

        self.input = (
            jnp.arange(len(charges))
            if not atom_type_embedding
            else (
                charges
                if subnet_type == 'mlp'
                else jnp.unique(charges, size=len(charges), return_inverse=True)[-1]
            )
        )
        if subnet_type == 'mlp':
            self.input = self.input[:, None]

    def __call__(self):
        return self.subnet(self.input)
