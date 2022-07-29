import haiku as hk
import jax.numpy as jnp
from jax import nn, ops
from jax.tree_util import tree_map

from ...hkext import MLP
from .distbasis import DistanceBasis
from .graph import Graph, GraphNodes, MessagePassingLayer


class PaiNNLayer(MessagePassingLayer):
    def __init__(
        self,
        name,
        ilayer,
        embedding_dim,
        dist_feat_dim,
        distance_basis,
        shared_h=True,
        shared_g=False,
        w_subnet=None,
        h_subnet=None,
        g_subnet=None,
        *,
        n_layers_w=1,
        n_layers_h=2,
        n_layers_g=2,
    ):
        super().__init__(name, ilayer)

        def default_subnet_kwargs(n_layers):
            return {
                'hidden_layers': ('log', n_layers),
                'last_bias': True,
                'last_linear': True,
                'activation': nn.silu,
            }

        self.labels = ['same', 'anti', 'ne', 'nn', 'en']
        self.w = {
            lbl: MLP(
                dist_feat_dim,
                3 * embedding_dim,
                name=f'w_{lbl}',
                **(w_subnet or default_subnet_kwargs(n_layers_w)),
            )
            for lbl in self.labels
        }
        self.h = (
            MLP(
                embedding_dim,
                3 * embedding_dim,
                name='h',
                **(h_subnet or default_subnet_kwargs(n_layers_h)),
            )
            if shared_h
            else {
                lbl: MLP(
                    embedding_dim,
                    3 * embedding_dim,
                    name=f'h_{lbl}',
                    **(h_subnet or default_subnet_kwargs(n_layers_h)),
                )
                for lbl in self.labels
            }
        )
        self.g = (
            MLP(
                2 * embedding_dim,
                3 * embedding_dim,
                name='g',
                **(g_subnet or default_subnet_kwargs(n_layers_g)),
            )
            if shared_g
            else {
                lbl: MLP(
                    2 * embedding_dim,
                    3 * embedding_dim,
                    name=f'g_{lbl}',
                    **(g_subnet or default_subnet_kwargs(n_layers_g)),
                )
                for lbl in self.labels
            }
        )
        self.distance_basis = distance_basis
        self.V = {
            lbl: hk.Linear(embedding_dim, with_bias=False, name=f'V_{lbl}')
            for lbl in self.labels
        }
        self.U = {
            lbl: hk.Linear(embedding_dim, with_bias=False, name=f'U_{lbl}')
            for lbl in self.labels
        }
        self.shared_h = shared_h
        self.shared_g = shared_g

    def get_update_edges_fn(self):
        def update_edges_fn(nodes, edges):
            expanded = edges._replace(
                data={
                    'distances': tree_map(
                        lambda dists: self.distance_basis(dists),
                        edges.data['distances'],
                    ),
                    'directions': edges.data['directions'],
                }
            )
            return expanded

        return update_edges_fn if self.ilayer == 0 else None

    def get_aggregate_edges_for_nodes_fn(self):
        def aggregate_edges_for_nodes_fn(nodes, edges):
            n_elec = nodes.electrons['s'].shape[-2]
            n_nuc = nodes.nuclei['s'].shape[-2]
            node_map = {
                'ne': (n_elec, nodes.nuclei),
                'nn': (n_nuc, nodes.nuclei),
                'en': (n_nuc, nodes.electrons),
                'same': (n_elec, nodes.electrons),
                'anti': (n_elec, nodes.electrons),
            }
            we = {lbl: self.w[lbl](edges.data['distances'][lbl]) for lbl in self.labels}
            hx = {
                lbl: (self.h if self.shared_h else self.h[lbl])(
                    node_map[lbl][1]['s'][edges.senders[lbl]]
                )
                for lbl in self.labels
            }
            phi_w = {
                edge_type: {
                    weh_lbl: weh
                    for weh_lbl, weh in zip(
                        ['s', 'vv', 'vs'], jnp.split(we * hx[edge_type], 3, axis=-1)
                    )
                }
                for edge_type, we in we.items()
            }
            z = {}
            for edge_type, fws in phi_w.items():
                z[edge_type] = {}
                n_nodes, nodes = node_map[edge_type]
                z[edge_type]['s'] = ops.segment_sum(
                    fws['s'], edges.receivers[edge_type], n_nodes
                )
                vv = fws['vv'][..., None] * nodes['v'][edges.senders[edge_type]]
                vs = (
                    fws['vs'][..., None]
                    * edges.data['directions'][edge_type][..., None, :]
                )
                z[edge_type]['v'] = ops.segment_sum(
                    vv + vs, edges.receivers[edge_type], n_nodes
                )
            return z

        return aggregate_edges_for_nodes_fn

    def get_update_nodes_fn(self):
        def update_nodes_fn(nodes, z):
            def linear_comb(W, v):
                return jnp.swapaxes(W(jnp.swapaxes(v, -2, -1)), -2, -1)

            Vv = {
                edge_type: linear_comb(self.V[edge_type], z_edge['v'])
                for edge_type, z_edge in z.items()
            }

            Uv = {
                edge_type: linear_comb(self.U[edge_type], z_edge['v'])
                for edge_type, z_edge in z.items()
            }
            gs = {
                edge_type: self.g[edge_type](
                    jnp.concatenate(
                        [z_edge['s'], jnp.linalg.norm(Vv[edge_type], axis=-1)], axis=-1
                    )
                )
                for edge_type, z_edge in z.items()
            }
            a = {
                edge_type: {
                    a_lbl: aa
                    for a_lbl, aa in zip(
                        ['ss', 'vv', 'sv'], jnp.split(g_edge, 3, axis=-1)
                    )
                }
                for edge_type, g_edge in gs.items()
            }

            node_updates = {}
            for edge_type, aa in a.items():
                node_updates[edge_type] = {}
                node_updates[edge_type]['v'] = Uv[edge_type] * aa['vv'][..., None]
                auv = aa['sv'] * jnp.einsum('pei,pei->pe', Uv[edge_type], Vv[edge_type])
                node_updates[edge_type]['s'] = auv + aa['ss']
            updated_nodes = nodes._replace(
                nuclei={
                    lbl: nodes.nuclei[lbl]
                    + sum(node_updates[edge_type][lbl] for edge_type in ['nn', 'en'])
                    for lbl in ['s', 'v']
                },
                electrons={
                    lbl: nodes.electrons[lbl]
                    + sum(
                        node_updates[edge_type][lbl]
                        for edge_type in ['ne', 'same', 'anti']
                    )
                    for lbl in ['s', 'v']
                },
            )
            return updated_nodes

        return update_nodes_fn


class PaiNN(hk.Module):
    def __init__(
        self,
        n_nuc,
        n_up,
        n_down,
        coords,
        embedding_dim,
        dist_feat_dim,
        n_interactions=1,
        cutoff=10.0,
        layer_kwargs=None,
    ):
        super().__init__('PaiNN')
        self.n_elec = n_up + n_down
        self.coords = coords
        self.spin_idxs = jnp.array(
            self.n_elec * [0] if n_up == n_down else n_up * [0] + n_down * [1]
        )
        self.nuclei_idxs = jnp.arange(n_nuc)
        self.X = hk.Embed(
            1 if n_up == n_down else 2, embedding_dim, name='ElectronicEmbedding'
        )
        self.Y = hk.Embed(n_nuc, embedding_dim, name='NuclearEmbedding')
        self.v_elec = jnp.zeros((self.n_elec, embedding_dim, 3))
        self.v_nuc = jnp.zeros((n_nuc, embedding_dim, 3))
        self.layers = [
            PaiNNLayer(
                'PaiNNLayer',
                i,
                embedding_dim,
                dist_feat_dim,
                DistanceBasis(dist_feat_dim, cutoff, envelope='nocusp')
                if i == 0
                else None,
                **(layer_kwargs or {}),
            )
            for i in range(n_interactions)
        ]

    @classmethod
    def required_edge_types(cls):
        return ['nn', 'en', 'ne', 'same', 'anti']

    def __call__(self, rs, graph_edges):
        def compute_edge_data(labels, positions):
            def dist(senders, receivers):
                return jnp.sqrt(((receivers - senders) ** 2).sum(axis=-1))

            def dist_direc(senders, receivers):
                eps = jnp.finfo(jnp.float32).eps
                d = dist(senders, receivers)
                return d, (receivers - senders) / jnp.where(
                    d[..., None] > eps, d[..., None], eps
                )

            data = {'distances': {}, 'directions': {}}
            for lbl, pos in zip(labels, positions):
                data['distances'][lbl], data['directions'][lbl] = dist_direc(
                    pos[0][graph_edges.senders[lbl]],
                    pos[1][graph_edges.receivers[lbl]],
                )
            return data

        nuc_embedding = self.Y(self.nuclei_idxs)
        elec_embedding = self.X(self.spin_idxs)
        graph = Graph(
            GraphNodes(
                {'s': nuc_embedding, 'v': self.v_nuc},
                {'s': elec_embedding, 'v': self.v_elec},
            ),
            graph_edges._replace(
                data=compute_edge_data(
                    ['nn', 'ne', 'en', 'same', 'anti'],
                    [
                        (self.coords, self.coords),
                        (self.coords, rs),
                        (rs, self.coords),
                        (rs, rs),
                        (rs, rs),
                    ],
                )
            ),
        )
        for layer in self.layers:
            graph = layer(graph)
        return jnp.concatenate(
            [
                graph.nodes.electrons['s'],
                graph.nodes.electrons['v'].reshape(self.n_elec, -1),
            ],
            axis=-1,
        )
