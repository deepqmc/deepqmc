import haiku as hk
import jax.numpy as jnp
from jax import ops

from ...hkext import MLP
from .distbasis import DistanceBasis
from .graph import Graph, GraphNodes, MessagePassingLayer, MolecularGraphEdgeBuilder


class DeepErwinLayer(MessagePassingLayer):
    def __init__(
        self,
        ilayer,
        n_nuc,
        n_up,
        n_down,
        embedding_dim,
        labels,
        elec_linear,
        shared_g=True,
        w_subnet=None,
        h_subnet=None,
        g_subnet=None,
        *,
        hidden_layers_w=(40, 40),
        hidden_layers_h=(40, 40),
        hidden_layers_g=(40,),
        w_init='deeperwin',
        h_activation=jnp.tanh,
        w_activation=jnp.tanh,
        g_activation=jnp.tanh,
    ):
        super().__init__('DeepErwinLayer', ilayer)

        def default_subnet_kwargs(hidden_layers):
            return {
                'hidden_layers': hidden_layers,
                'bias': 'all',
                'w_init': w_init,
            }

        self.h = {
            lbl: MLP(
                -1,  # we do not use log hidden layers, so not needed
                embedding_dim,
                name=f'h_{lbl}',
                last_linear=False,
                activation=h_activation,
                **(h_subnet or default_subnet_kwargs(hidden_layers_h)),
            )
            for lbl in labels[:2]
        }
        self.w = {
            lbl: MLP(
                -1,  # we do not use log hidden layers, so not needed
                embedding_dim,
                name=f'w_{lbl}',
                last_linear=True,
                activation=w_activation,
                **(w_subnet or default_subnet_kwargs(hidden_layers_w)),
            )
            for lbl in labels
        }
        self.g = (
            MLP(
                embedding_dim,
                embedding_dim,
                name='g',
                last_linear=False,
                **(g_subnet or default_subnet_kwargs(hidden_layers_g)),
            )
            if shared_g
            else {
                lbl: MLP(
                    embedding_dim,
                    embedding_dim,
                    name=f'g_{lbl}',
                    last_linear=False,
                    activation=g_activation,
                    **(g_subnet or default_subnet_kwargs(hidden_layers_g)),
                )
                for lbl in labels
            }
        )
        self.labels = labels
        self.shared_g = shared_g
        self.elec_linear = elec_linear

    def get_update_edges_fn(self):
        return None

    def get_aggregate_edges_for_nodes_fn(self):
        def aggregate_edges_for_nodes_fn(nodes, edges):
            n_elec = nodes.electrons.shape[-2]
            if not self.elec_linear and self.ilayer == 0:
                hx_same, hx_anti = (
                    nodes.electrons[edges['same'].senders],
                    nodes.electrons[edges['anti'].senders],
                )
            else:
                hx_same, hx_anti = (
                    self.h[lbl](nodes.electrons)[edges[lbl].senders]
                    for lbl in self.labels[:2]
                )
            hx_ne = nodes.nuclei[edges['ne'].senders]
            we_same, we_anti, we_n = (
                self.w[lbl](edges[lbl].data['features']) for lbl in self.labels
            )
            weh_same = we_same * hx_same
            weh_anti = we_anti * hx_anti
            weh_n = we_n * hx_ne
            z_same = ops.segment_sum(
                data=weh_same, segment_ids=edges['same'].receivers, num_segments=n_elec
            )
            z_anti = ops.segment_sum(
                data=weh_anti, segment_ids=edges['anti'].receivers, num_segments=n_elec
            )
            z_n = ops.segment_sum(
                data=weh_n, segment_ids=edges['ne'].receivers, num_segments=n_elec
            )
            return {
                'same': z_same,
                'anti': z_anti,
                'ne': z_n,
            }

        return aggregate_edges_for_nodes_fn

    def get_update_nodes_fn(self):
        def update_nodes_fn(nodes, z):
            if self.shared_g:
                z_all = sum(z.values())
                updated_nodes = nodes._replace(electrons=self.g(z_all))
            else:
                updated_nodes = nodes._replace(
                    electrons=sum(self.g[lbl](z[lbl]) for lbl in self.labels)
                )
            return updated_nodes

        return update_nodes_fn


class DeepErwinEmbedding(hk.Module):
    def __init__(
        self,
        n_nuc,
        n_up,
        n_down,
        coords,
        embedding_dim,
        dist_feat_dim=32,
        n_interactions=3,
        envelope='nocusp',
        layer_kwargs=None,
        ghost_coords=None,
        use_rbf_features=True,
        distance_feature_powers=None,
        use_el_ion_differences=False,
        use_el_el_differences=False,
        Z=None,
        nuc_linear=True,
        elec_linear=True,
    ):
        super().__init__('DeepErwinEmbedding')
        labels = ['same', 'anti', 'ne']
        if ghost_coords is not None:
            n_nuc += len(ghost_coords)
            coords = jnp.concatenate([coords, jnp.asarray(ghost_coords)])
        self.coords = coords
        dist_basis = DistanceBasis(dist_feat_dim, 5.0, envelope=envelope, offset=False)
        self.n_nuc, self.n_up, self.n_down = n_nuc, n_up, n_down

        def common_callback(pos_sender, pos_receiver, sender_idx, receiver_idx):
            differences = pos_receiver[receiver_idx] - pos_sender[sender_idx]
            distances = jnp.linalg.norm(differences, axis=-1)
            features = []
            if use_rbf_features:
                features.append(dist_basis(distances))
            if distance_feature_powers:
                eps = 1e-2
                features_dist = jnp.stack(
                    [
                        distances**n if n > 0 else 1 / (distances ** (-n) + eps)
                        for n in distance_feature_powers
                    ],
                    axis=-1,
                )
                features.append(features_dist)
            return features, differences

        def ne_callback(pos_sender, pos_receiver, sender_idx, receiver_idx):
            features, differences = common_callback(
                pos_sender, pos_receiver, sender_idx, receiver_idx
            )
            if use_el_ion_differences:
                features.append(differences)
            assert features
            return {'features': jnp.concatenate(features, axis=-1)}

        def el_callback(pos_sender, pos_receiver, sender_idx, receiver_idx):
            features, differences = common_callback(
                pos_sender, pos_receiver, sender_idx, receiver_idx
            )
            if use_el_el_differences:
                features.append(differences)
            assert features
            return {'features': jnp.concatenate(features, axis=-1)}

        self.edge_factory = MolecularGraphEdgeBuilder(
            n_nuc,
            n_up,
            n_down,
            coords,
            labels,
            kwargs_by_edge_type={
                'ne': {'cutoff': 100.0, 'data_callback': ne_callback},
                'same': {'cutoff': 100.0, 'data_callback': el_callback},
                'anti': {'cutoff': 100.0, 'data_callback': el_callback},
            },
        )
        if nuc_linear:
            self.nuc_idxs = jnp.asarray(Z, dtype=jnp.float32)[..., None]
            self.Y = MLP(
                1,
                embedding_dim,
                name='ion_emb',
                hidden_layers=[],
                bias='all',
                last_linear=False,
                activation=jnp.tanh,
                w_init='deeperwin',
            )
        else:
            self.nuc_idxs = jnp.arange(n_nuc)
            self.Y = hk.Embed(n_nuc, embedding_dim, name='ion_emb')
        if elec_linear:
            elec_idx = jnp.array(
                (n_up + n_down) * [0] if n_up == n_down else n_up * [0] + n_down * [1]
            )
            self.spin_idxs = jnp.eye(1 if n_up == n_down else 2)[elec_idx]
        else:
            self.spin_idxs = jnp.array(
                (n_up + n_down) * [0] if n_up == n_down else n_up * [0] + n_down * [1]
            )
            self.X = hk.Embed(
                1 if n_up == n_down else 2, embedding_dim, name='ElectronicEmbedding'
            )
        self.elec_linear = elec_linear

        self.layers = [
            DeepErwinLayer(
                i,
                n_nuc,
                n_up,
                n_down,
                embedding_dim,
                labels,
                elec_linear,
                **(layer_kwargs or {}),
            )
            for i in range(n_interactions)
        ]

    def __call__(self, r):
        def init_state(shape, dtype):
            def zeros(l):
                return jnp.zeros(max(1, l), dtype)

            return {
                'anti': (
                    zeros(self.n_up * self.n_down),
                    zeros(self.n_up * self.n_down),
                ),
                'ne': zeros(self.n_nuc * (self.n_up + self.n_down)),
                'same': (
                    zeros(self.n_up * (self.n_up - 1)),
                    zeros(self.n_down * (self.n_down - 1)),
                ),
            }

        occupancies = hk.get_state(
            'occupancies',
            shape=1,
            dtype=jnp.int32,
            init=init_state,
        )
        n_occupancies = hk.get_state(
            'n_occupancies', shape=[], dtype=jnp.int32, init=jnp.zeros
        )
        graph_edges, occupancies, n_occupancies = self.edge_factory(
            r, occupancies, n_occupancies
        )
        hk.set_state('occupancies', occupancies)
        hk.set_state('n_occupancies', n_occupancies)
        graph = Graph(
            GraphNodes(
                self.Y(self.nuc_idxs),
                self.spin_idxs if self.elec_linear else self.X(self.spin_idxs),
            ),
            graph_edges,
        )

        for layer in self.layers:
            graph = layer(graph)

        return graph.nodes.electrons
