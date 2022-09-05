import haiku as hk
import jax.numpy as jnp
from jax import nn, ops

from ...hkext import MLP
from .distbasis import DistanceBasis
from .graph import (
    Graph,
    GraphNodes,
    MessagePassingLayer,
    MolecularGraphEdgeBuilder,
    distance_direction_callback,
)


class PaiNNLayer(MessagePassingLayer):
    def __init__(
        self,
        ilayer,
        n_nuc,
        n_up,
        n_down,
        embedding_dim,
        dist_feat_dim,
        distance_basis,
        labels,
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
        super().__init__('PaiNNLayer', ilayer)

        def default_subnet_kwargs(n_layers):
            return {
                'hidden_layers': ('log', n_layers),
                'bias': False,
                'last_linear': True,
                'activation': nn.silu,
            }

        self.w = {
            lbl: MLP(
                dist_feat_dim,
                3 * embedding_dim,
                name=f'w_{lbl}',
                **(w_subnet or default_subnet_kwargs(n_layers_w)),
            )
            for lbl in labels
        }
        self.spin_idxs = jnp.array(
            (n_up + n_down) * [0] if n_up == n_down else n_up * [0] + n_down * [1]
        )
        self.nuc_idxs = jnp.arange(n_nuc)
        self.h = {
            lbl: hk.Embed(
                1 if n_up == n_down else 2, 3 * embedding_dim, name=f'h_{lbl}'
            )
            if self.ilayer == 0
            else MLP(
                embedding_dim,
                3 * embedding_dim,
                name=f'h_{lbl}',
                **(h_subnet or default_subnet_kwargs(n_layers_h)),
            )
            for lbl in labels
        }
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
                for lbl in labels
            }
        )
        self.distance_basis = distance_basis
        self.V = {
            lbl: hk.Linear(
                embedding_dim,
                with_bias=False,
                name=f'V_{lbl}',
                w_init=lambda shape, dtype: jnp.eye(shape[0], dtype=dtype),
            )
            for lbl in labels
        }
        self.U = {
            lbl: hk.Linear(
                embedding_dim,
                with_bias=False,
                name=f'U_{lbl}',
                w_init=lambda shape, dtype: jnp.eye(shape[0], dtype=dtype),
            )
            for lbl in labels
        }
        self.labels = labels
        self.shared_h = shared_h
        self.shared_g = shared_g

    def get_update_edges_fn(self):
        def update_edges_fn(nodes, edges):
            return {
                k: edge._replace(
                    data={
                        'distances': self.distance_basis(edge.data['distances']),
                        'directions': edge.data['directions'],
                    }
                )
                for k, edge in edges.items()
            }

        return update_edges_fn if self.ilayer == 0 else None

    def get_aggregate_edges_for_nodes_fn(self):
        def aggregate_edges_for_nodes_fn(nodes, edges):
            n_elec = nodes.electrons['s'].shape[-2]
            n_nuc = nodes.nuclei['s'].shape[-2]
            node_map = {
                'ne': (n_elec, nodes.nuclei, self.nuc_idxs),
                'en': (n_nuc, nodes.electrons, self.spin_idxs),
                'same': (n_elec, nodes.electrons, self.spin_idxs),
                'anti': (n_elec, nodes.electrons, self.spin_idxs),
            }
            we = {lbl: self.w[lbl](edges[lbl].data['distances']) for lbl in self.labels}
            hx = {
                lbl: self.h[lbl](
                    node_map[lbl][2] if self.ilayer == 0 else node_map[lbl][1]['s']
                )[edges[lbl].senders]
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
                n_nodes, nodes, _ = node_map[edge_type]
                z[edge_type]['s'] = ops.segment_sum(
                    fws['s'], edges[edge_type].receivers, n_nodes
                )
                vv = fws['vv'][..., None] * nodes['v'][edges[edge_type].senders]
                vs = (
                    fws['vs'][..., None]
                    * edges[edge_type].data['directions'][..., None, :]
                )
                z[edge_type]['v'] = ops.segment_sum(
                    vv + vs, edges[edge_type].receivers, n_nodes
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

            def norm(x):
                shape = x.shape[:-1]
                norm = jnp.linalg.norm(x, axis=-1).flatten()
                norm_p1 = jnp.concatenate([norm, jnp.array([1e-7])])
                mask = norm > 1e-5
                idx = jnp.where(mask, jnp.arange(len(norm)), jnp.array(len(norm)))
                norm = norm_p1[idx]
                return norm.reshape(shape)

            gs = {
                edge_type: self.g[edge_type](
                    jnp.concatenate(
                        [z_edge['s'], (Vv[edge_type] ** 2).sum(axis=-1)],
                        axis=-1,
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
                    + sum(node_updates[edge_type][lbl] for edge_type in ['en'])
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
        ghost_coords=None,
    ):
        super().__init__('PaiNN')
        assert embedding_dim % 4 == 0
        _embedding_dim = embedding_dim // 4
        labels = ['en', 'ne', 'same', 'anti']
        self.n_elec = n_up + n_down
        if ghost_coords is not None:
            n_nuc += len(ghost_coords)
            coords = jnp.concatenate([coords, jnp.asarray(ghost_coords)])
        self.coords = coords
        self.edge_factory = MolecularGraphEdgeBuilder(
            n_nuc,
            n_up,
            n_down,
            coords,
            labels,
            kwargs_by_edge_type={
                lbl: {'cutoff': cutoff, 'data_callback': distance_direction_callback}
                for lbl in labels
            },
        )
        self.spin_idxs = jnp.array(
            self.n_elec * [0] if n_up == n_down else n_up * [0] + n_down * [1]
        )
        self.X = hk.Embed(
            1 if n_up == n_down else 2, _embedding_dim, name='ElectronicEmbedding'
        )
        self.nuclei_idxs = jnp.arange(n_nuc)
        self.Y = hk.Embed(n_nuc, _embedding_dim, name='NuclearEmbedding')
        self.v_elec = jnp.zeros((self.n_elec, _embedding_dim, 3))
        self.v_nuc = jnp.zeros((n_nuc, _embedding_dim, 3))
        self.layers = [
            PaiNNLayer(
                i,
                n_nuc,
                n_up,
                n_down,
                _embedding_dim,
                dist_feat_dim,
                DistanceBasis(dist_feat_dim, cutoff, envelope='nocusp')
                if i == 0
                else None,
                labels,
                **(layer_kwargs or {}),
            )
            for i in range(n_interactions)
        ]

    def init_state(self, shape, dtype):
        zeros = jnp.zeros(shape, dtype)
        return {
            'en': zeros,
            'ne': zeros,
            'same': (zeros, zeros),
            'anti': (zeros, zeros),
        }

    def __call__(self, rs):
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
            rs, occupancies, n_occupancies
        )
        hk.set_state('occupancies', occupancies)
        hk.set_state('n_occupancies', n_occupancies)

        nuc_embedding = self.Y(self.nuclei_idxs)
        elec_embedding = self.X(self.spin_idxs)
        graph = Graph(
            GraphNodes(
                {'s': nuc_embedding, 'v': self.v_nuc},
                {'s': elec_embedding, 'v': self.v_elec},
            ),
            graph_edges,
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
