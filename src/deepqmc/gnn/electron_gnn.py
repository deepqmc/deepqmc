from functools import partial
from itertools import accumulate

import haiku as hk
import jax.numpy as jnp
from jax import tree_util

from ..hkext import MLP
from .graph import Graph, GraphNodes, GraphUpdate, MolecularGraphEdgeBuilder
from .utils import NodeEdgeMapping


class ElectronGNNLayer(hk.Module):
    r"""
    The message passing layer of :class:`ElectronGNN`.

    Derived from :class:`~deepqmc.gnn.gnn.MessagePassingLayer`.

    Args:
        one_particle_residual: whether a residual connection is used when updating
            the one particle embeddings, either :data:`False`, or an instance of
            :class:`~deepqmc.hkext.ResidualConnection`.
        two_particle_residual: whether a residual connection is used when updating
            the two particle embeddings, either :data:`False`, or an instance of
            :class:`~deepqmc.hkext.ResidualConnection`.
        deep_features: if :data:`False`, the edge features are not updated throughout
            the GNN layers, if :data:`shared` than in each layer a single MLP
            (:data:`u`) is used to update all edge types, if :data:`separate` then in
            each layer separate MLPs are used to update the different edge types.
        update_features (List[~deepqmc.gnn.update_features.UpdateFeature]): a list of
            partially initialized update feature classes to use when computing the
            update features of the one particle embeddings. For more details see the
            documentation of :class:`deepqmc.gnn.update_features`.
        update_rule (str): how to combine the update features for the update of the
            one particle embeddings.
            Possible values:

            - ``'concatenate'``: run concatenated features through MLP
            - ``'featurewise'``: apply different MLP to each feature channel and sum
            - ``'featurewise_shared'``: apply the same MLP across feature channels
            - ``'sum'``: sum features before sending through an MLP

            note that :data:`'sum'` and :data:`'featurewise_shared'` imply features
            of same size.
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
        two_particle_stream_dim,
        *,
        one_particle_residual,
        two_particle_residual,
        deep_features,
        update_features,
        update_rule,
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
        assert (
            update_rule not in ['sum', 'featurewise_shared']
            or embedding_dim == two_particle_stream_dim
        )
        assert deep_features in [False, 'shared', 'separate']
        self.deep_features = deep_features
        self.update_rule = update_rule
        subnet_factory_by_lbl = subnet_factory_by_lbl or {}
        for lbl in ['g', 'u']:
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
        self.update_features = [
            uf(self.n_up, self.n_down, two_particle_stream_dim, self.mapping)
            for uf in update_features
        ]
        self.g = (
            subnet_factory_by_lbl['g'](
                embedding_dim,
                name='g',
            )
            if not self.update_rule == 'featurewise'
            else {
                name: subnet_factory_by_lbl['g'](
                    embedding_dim,
                    name=f'g_{name}',
                )
                for uf in (self.update_features)
                for name in uf.names
            }
        )
        self.one_particle_residual = one_particle_residual
        self.two_particle_residual = two_particle_residual
        self.self_interaction = self_interaction

    def get_update_edges_fn(self):
        def update_edges(edges):
            if self.deep_features:
                if self.deep_features == 'shared':
                    # combine features along leading dim, apply MLP and split
                    # into channels again to please kfac
                    keys, edge_objects = zip(*edges.items())
                    feats = [e.single_array for e in edge_objects]
                    split_idxs = list(accumulate(len(f) for f in feats))
                    feats = jnp.split(self.u(jnp.concatenate(feats)), split_idxs)
                    edge_objects = [
                        e.update_from_single_array(f)
                        for e, f in zip(edge_objects, feats)
                    ]
                    updated_edges = dict(zip(keys, edge_objects))
                elif self.deep_features == 'separate':
                    updated_edges = {
                        typ: edge.update_from_single_array(
                            self.u[typ](edge.single_array)
                        )
                        for typ, edge in edges.items()
                    }

                if self.two_particle_residual:
                    updated_edges = self.two_particle_residual(edges, updated_edges)
                return updated_edges
            else:
                return edges

        return update_edges

    def get_aggregate_edges_for_nodes_fn(self):
        def aggregate_edges_for_nodes(nodes, edges):
            f = []
            for uf in self.update_features:
                f.extend(uf(nodes, edges))
            return f

        return aggregate_edges_for_nodes

    def get_update_nodes_fn(self):
        def update_nodes(nodes, f):
            if self.update_rule == 'concatenate':
                updated = self.g(jnp.concatenate(f, axis=-1))
            elif self.update_rule == 'featurewise':
                updated = sum(self.g[name](fi) for fi, name in zip(f, self.g.keys()))
            elif self.update_rule == 'sum':
                updated = self.g(sum(f))
            elif self.update_rule == 'featurewise_shared':
                updated = jnp.sum(self.g(jnp.stack(f)), axis=0)
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
        hamil (:class:`~deepqmc.MolecularHamiltonian`): the Hamiltonian of the system
            on which the graph is defined.
        embedding_dim (int): the length of the electron embedding vectors.
        n_interactions (int): number of message passing interactions.
        edge_features (dict): a :data:`dict` of functions for each edge
            type, embedding the interparticle differences. Valid keys are:

            - ``'ne'``: for nucleus-electron edges
            - ``'nn'``: for nucleus-nucleus edges
            - ``'same'``: for same spin electron-electron edges
            - ``'anti'``: for opposite spin electron-electron edges
            - ``'up'``: for edges going from spin up electrons to all electrons
            - ``'down'``: for edges going from spin down electrons to all electrons

        self_interaction (bool): whether to consider edges where the sender and
            receiver electrons are the same.
        two_particle_stream_dim (int): the feature dimension of the two particle
            streams. Only active if :data:`deep_features` are used.
        nuclei_embedding (Union[None,~deepqmc.gnn.electron_gnn.NucleiEmbedding]):
            optional, the instance responsible for creating the initial nuclear
            embeddings. Set to :data:`None` if nuclear embeddings are not needed.
        electron_embedding (~deepqmc.gnn.electron_gnn.ElectronEmbedding): the instance
            that creates the initial electron embeddings.
        layer_factory (Callable): a callable that generates a layer of the GNN.
        ghost_coords: optional, specifies the coordinates of one or more ghost atoms,
            useful for breaking spatial symmetries of the nuclear geometry.
    """

    def __init__(
        self,
        hamil,
        embedding_dim,
        *,
        n_interactions,
        edge_features,
        self_interaction,
        two_particle_stream_dim,
        nuclei_embedding,
        electron_embedding,
        layer_factory,
        ghost_coords=None,
    ):
        super().__init__()
        n_nuc, n_up, n_down = hamil.n_nuc, hamil.n_up, hamil.n_down
        n_atom_types = hamil.mol.n_atom_types
        charges = hamil.mol.charges
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
        self.edge_types = tuple((edge_features or {}).keys())
        self.layers = [
            layer_factory(
                n_interactions,
                ilayer,
                n_nuc,
                n_up,
                n_down,
                embedding_dim,
                self.edge_types,
                self_interaction,
                self.node_data,
                two_particle_stream_dim,
            )
            for ilayer in range(n_interactions)
        ]
        self.edge_features = edge_features
        self.nuclei_embedding = (
            nuclei_embedding(charges, n_atom_types) if nuclei_embedding else None
        )
        self.electron_embedding = electron_embedding(
            n_nuc,
            n_up,
            n_down,
            embedding_dim,
            self.node_data['n_node_types']['electrons'],
            self.node_data['node_types']['electrons'],
        )
        self.self_interaction = self_interaction

    def node_factory(self, phys_conf):
        electron_embedding = self.electron_embedding(phys_conf)
        nucleus_embedding = self.nuclei_embedding() if self.nuclei_embedding else None
        return GraphNodes(nucleus_embedding, electron_embedding)

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
        return {
            typ: edges[typ].update_from_single_array(
                self.edge_features[typ](edges[typ].single_array)
            )
            for typ in self.edge_types
        }

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


class ElectronEmbedding(hk.Module):
    r"""Create initial embeddings for electrons.

    Args:
        n_nuc (int): the number of nuclei.
        n_up (int): the number of spin up electrons.
        n_down (int): the number of spin down electrons.
        embedding_dim (int): the desired length of the embedding vectors.
        n_elec_types (int): the number of electron types to differentiate.
            Usual values are:

            - ``1``: treat all electrons as indistinguishable. Note that electrons
                with different spins can still become distinguishable during the later
                embedding update steps of the GNN.
            - ``2``: treat spin up and spin down electrons as distinguishable already
                in the initial embeddings.

        elec_types (jax.Array): an integer array with length equal to the number of
            electrons, with entries between ``0`` and ``n_elec_types``. Specifies the
            type for each electron.
        positional_embeddings (Union[Literal[False],dict]): if not ``False``, a
            ``dict`` with edge types as keys, and edge features as values. Specifies
            the edge types and edge features to use when constructing the positional
            initial electron embeddings.
        use_spin (bool): only relevant if ``positional_embeddings`` is not ``False``,
            if ``True``, concatenate the spin of the given electron after the
            positional embedding features.
        project_to_embedding_dim (bool): only relevant if ``positional_embeddings``
            is not ``False``, if ``True``, use a linear layer to project the initial
            embeddings to have length ``embedding_dim``.
    """

    def __init__(
        self,
        n_nuc,
        n_up,
        n_down,
        embedding_dim,
        n_elec_types,
        elec_types,
        *,
        positional_embeddings,
        use_spin,
        project_to_embedding_dim,
    ):
        super().__init__()
        self.n_nuc = n_nuc
        self.n_up = n_up
        self.n_down = n_down
        self.embedding_dim = embedding_dim
        self.n_elec_types = n_elec_types
        self.elec_types = elec_types
        self.positional_embeddings = positional_embeddings
        self.use_spin = use_spin
        self.project_to_embedding_dim = project_to_embedding_dim

    def __call__(self, phys_conf):
        if self.positional_embeddings:
            edge_factory = MolecularGraphEdgeBuilder(
                self.n_nuc,
                self.n_up,
                self.n_down,
                self.positional_embeddings.keys(),
                self_interaction=False,
            )
            feats = tree_util.tree_map(
                lambda f, e: f(e.single_array)
                .swapaxes(0, 1)
                .reshape(self.n_up + self.n_down, -1),
                self.positional_embeddings,
                edge_factory(phys_conf),
            )
            x = tree_util.tree_reduce(partial(jnp.concatenate, axis=1), feats)
            if self.use_spin:
                spins = jnp.concatenate([jnp.ones(self.n_up), -jnp.ones(self.n_down)])[
                    :, None
                ]
                x = jnp.concatenate([x, spins], axis=1)
            if self.project_to_embedding_dim:
                x = hk.Linear(self.embedding_dim, with_bias=False)(x)
        else:
            X = hk.Embed(
                self.n_elec_types, self.embedding_dim, name='ElectronicEmbedding'
            )
            x = X(self.elec_types)
        return x
