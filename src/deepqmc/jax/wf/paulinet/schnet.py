import haiku as hk
import jax.numpy as jnp
from jax import ops

from deepqmc.jax.hkext import SSP
from .distbasis import DistanceBasis
from .graph import GraphBuilder, GraphNetwork


class SchNetLayer(hk.Module):
    def __init__(
        self,
        embedding_dim,
        kernel_dim,
        dist_feat_dim,
        shared_h=True,
        shared_g=False,
    ):
        super().__init__('SchNetLayer')

        labels = ['same', 'anti', 'n']
        self.w = {
            lbl: hk.nets.MLP(
                [dist_feat_dim, kernel_dim], activation=SSP, name=f'w_{lbl}'
            )
            for lbl in labels
        }
        self.h = (
            hk.nets.MLP([kernel_dim], activation=SSP, name='h')
            if shared_h
            else {
                lbl: hk.nets.MLP([kernel_dim], activation=SSP, name=f'h_{lbl}')
                for lbl in labels
            }
        )
        self.g = (
            hk.nets.MLP([embedding_dim], activation=SSP)
            if shared_g
            else {
                lbl: hk.nets.MLP([embedding_dim], activation=SSP, name=f'g_{lbl}')
                for lbl in labels
            }
        )
        self.kernel_dim = kernel_dim
        self.labels = labels
        self.shared_h = shared_h
        self.shared_g = shared_g

        self.forward = GraphNetwork(
            update_node_fn=self.get_update_node_fn(),
            aggregate_edges_for_nodes_fn=self.get_aggregate_edges_for_nodes_fn(),
        )

    def __call__(self, graph):
        return self.forward(graph)

    def get_aggregate_edges_for_nodes_fn(self):
        def aggregate_edges_for_nodes_fn(nodes, edges, senders, receivers, n_nodes):
            dist, e_typ = edges['dist'], edges['type'][..., None]
            zeros = jnp.zeros_like(
                dist, shape=(*dist.shape[:-1], nodes['nuc'].shape[-1])
            )
            nuc_or_zero = jnp.where(e_typ == 1, self.w['n'](dist), zeros)
            anti_or_nuc = jnp.where(e_typ == 4, self.w['anti'](dist), nuc_or_zero)
            we = jnp.where(e_typ == 3, self.w['same'](dist), anti_or_nuc)
            hx = jnp.concatenate(
                [nodes['nuc'], self.h(nodes['elec'])], axis=-2
            ).reshape(-1, nodes['nuc'].shape[-1])
            weh = we * hx[senders]
            weh_same = jnp.where(e_typ == 3, weh, zeros)
            weh_anti = jnp.where(e_typ == 4, weh, zeros)
            weh_nuc = jnp.where(e_typ == 1, weh, zeros)
            z_same = ops.segment_sum(
                data=weh_same, segment_ids=receivers, num_segments=n_nodes
            )
            z_anti = ops.segment_sum(
                data=weh_anti, segment_ids=receivers, num_segments=n_nodes
            )
            z_nuc = ops.segment_sum(
                data=weh_nuc, segment_ids=receivers, num_segments=n_nodes
            )
            return {
                'nuc': z_nuc,
                'same': z_same,
                'anti': z_anti,
            }

        return aggregate_edges_for_nodes_fn

    def get_update_node_fn(self):
        def update_node_fn(nodes, z):
            def eval_g(g, message):
                n_nuclei = nodes['nuc'].shape[-2]
                n_particles = nodes['elec'].shape[-2] + n_nuclei
                return g(
                    message.reshape(-1, n_particles, self.kernel_dim)[:, n_nuclei:],
                )

            nodes['elec'] += (
                eval_g(self.g if self.shared_g else self.g['n'], z['nuc'])
                + eval_g(self.g if self.shared_g else self.g['same'], z['same'])
                + eval_g(self.g if self.shared_g else self.g['anti'], z['anti'])
            )
            return nodes

        return update_node_fn


class SchNet(hk.Module):
    def __init__(
        self,
        n_nuc,
        n_up,
        n_down,
        coords,
        graph_builder,
        embedding_dim,
        dist_feat_dim,
        kernel_dim=128,
        n_interactions=2,
        cutoff=10.0,
        initial_occupancy=10,
    ):
        super().__init__('SchNet')
        self.coords = coords
        elec_vocab_size = 1 if n_up == n_down else 2
        spin_idxs = jnp.array(
            (n_up + n_down) * [0] if n_up == n_down else n_up * [0] + n_down * [1]
        )
        self.X = hk.Embed(elec_vocab_size, embedding_dim, name='ElectronicEmbedding')
        self.Y = hk.Embed(n_nuc, kernel_dim, name='NuclearEmbedding')
        self.spin_idxs = spin_idxs
        self.nuclei_idxs = jnp.arange(n_nuc)
        self.graph_builder = graph_builder
        self.layers = [
            SchNetLayer(embedding_dim, kernel_dim, dist_feat_dim)
            for _ in range(n_interactions)
        ]

    def neighbor_list_from_rs(self, rs):
        batch_dims = rs.shape[:-2]
        positions = jnp.concatenate(
            [
                jnp.tile(
                    jnp.expand_dims(self.coords, jnp.arange(len(batch_dims))),
                    (*batch_dims, 1, 1),
                ),
                rs,
            ],
            axis=-2,
        )
        return self.neighbor_list_builder(positions)

    def __call__(self, neighbor_list):
        nuc_embedding = self.Y(self.nuclei_idxs)
        elec_embedding = self.X(self.spin_idxs)
        graph = self.graph_builder(neighbor_list, nuc_embedding, elec_embedding)
        for layer in self.layers:
            graph = layer(graph)
        return graph.nodes['elec']

    @classmethod
    def from_mol(cls, rng, mol, nl, *args, **kwargs):
        n_nuc, n_up, n_down = mol.n_particles()
        graph_builder = GraphBuilder(
            n_nuc,
            n_up,
            n_down,
            10.0,
            DistanceBasis(16, envelope='nocusp'),
        )

        @hk.without_apply_rng
        @hk.transform
        def schnet_fwd(neighbor_list):
            return cls(
                n_nuc,
                n_up,
                n_down,
                mol.coords,
                graph_builder,
                embedding_dim=64,
                cutoff=10.0,
            )(neighbor_list)

        params = schnet_fwd.init(rng, nl)
        return params, schnet_fwd.apply
