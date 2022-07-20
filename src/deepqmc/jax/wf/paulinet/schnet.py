from functools import partial

import haiku as hk
import jax.numpy as jnp
from jax import ops, random

from deepqmc.jax.wf.paulinet.distbasis import DistanceBasis
from deepqmc.jax.wf.paulinet.graph import GraphBuilder, GraphNetwork


def subnet(rng, inp_dim, out_dim):
    def _subnet_forward(x):
        subnet = hk.nets.MLP([10, out_dim])
        return subnet(x)

    _subnet = hk.without_apply_rng(hk.transform(_subnet_forward))
    params = _subnet.init(rng, jnp.zeros((1, inp_dim)))
    net = partial(_subnet.apply, params)
    return net, params


class SchNetLayer:
    def __init__(self, rng, embedding_dim, kernel_dim, dist_feat_dim):
        self.kernel_dim = kernel_dim
        (
            rng_w_nuc,
            rng_w_same,
            rng_w_anti,
            rng_h,
            rng_g_nuc,
            rng_g_same,
            rng_g_anti,
        ) = random.split(rng, 7)
        self.w_nuc, self.w_nuc_params = subnet(rng_w_nuc, dist_feat_dim, kernel_dim)
        self.w_same, self.w_same_params = subnet(rng_w_same, dist_feat_dim, kernel_dim)
        self.w_anti, self.w_anti_params = subnet(rng_w_anti, dist_feat_dim, kernel_dim)
        self.h, self.h_params = subnet(rng_h, embedding_dim, kernel_dim)
        self.g_nuc, self.g_nuc_params = subnet(rng_g_nuc, kernel_dim, embedding_dim)
        self.g_same, self.g_same_params = subnet(rng_g_same, kernel_dim, embedding_dim)
        self.g_anti, self.g_anti_params = subnet(rng_g_anti, kernel_dim, embedding_dim)

    def aggregate_edges_for_nodes_fn(self, nodes, edges, senders, receivers, n_nodes):
        dist, e_typ = edges['dist'], edges['type'][:, None]
        zeros = jnp.zeros_like(dist, shape=(*dist.shape[:-1], self.kernel_dim))
        nuc_or_zero = jnp.where(e_typ == 1, self.w_nuc(dist), zeros)
        anti_or_nuc = jnp.where(e_typ == 4, self.w_anti(dist), nuc_or_zero)
        we = jnp.where(e_typ == 3, self.w_same(dist), anti_or_nuc)
        hx = jnp.concatenate([nodes['nuc'], self.h(nodes['elec'])], axis=-2).reshape(
            -1, self.kernel_dim
        )
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

    def update_node_fn(self, nodes, z):
        def eval_g(g, message):
            n_nuclei = nodes['nuc'].shape[-2]
            n_particles = nodes['elec'].shape[-2] + n_nuclei
            return g(message.reshape(-1, n_particles, self.kernel_dim)[:, n_nuclei:])

        nodes['elec'] += (
            eval_g(self.g_nuc, z['nuc'])
            + eval_g(self.g_same, z['same'])
            + eval_g(self.g_anti, z['anti'])
        )
        return nodes


class SchNet:
    def __init__(
        self,
        rng,
        mol,
        embedding_dim,
        kernel_dim,
        dist_feat_dim,
        n_interactions=1,
        cutoff=10.0,
    ):
        n_nuclei = len(mol.charges)
        n_electrons = int(mol.charges.sum()) - mol.charge
        n_up = (n_electrons + mol.spin) // 2
        n_down = n_electrons - n_up
        self.mol = mol

        rng, rng_rs = random.split(rng)
        rs = random.normal(rng_rs, (n_electrons, 3))
        positions = jnp.concatenate([mol.coords, rs])
        self.graph_builder = GraphBuilder(
            n_nuclei,
            n_up,
            n_down,
            cutoff,
            positions,
            DistanceBasis(dist_feat_dim),
            #  n_nuclei + jnp.zeros(n_electrons)[:, None],
            jnp.tile(n_nuclei + jnp.arange(n_electrons)[:, None], (1, embedding_dim)),
            jnp.tile(jnp.arange(n_nuclei)[:, None], (1, kernel_dim)),
            #  jnp.tile(jnp.array(range(n_nuclei-1,-1,-1))[:, None], (1, kernel_dim)),
            #  jnp.tile(jnp.zeros(n_nuclei)[:, None], (1, kernel_dim)),
        )
        rng, *rng_layers = random.split(rng, n_interactions + 1)
        self.layers = [
            SchNetLayer(rng_layer, embedding_dim, kernel_dim, dist_feat_dim)
            for rng_layer in rng_layers
        ]
        self.apply_layers = [
            GraphNetwork(
                update_node_fn=layer.update_node_fn,
                aggregate_edges_for_nodes_fn=layer.aggregate_edges_for_nodes_fn,
            )
            for layer in self.layers
        ]

    def __call__(self, rs):
        batch_dims = rs.shape[:-2]
        positions = jnp.concatenate(
            [
                jnp.tile(
                    jnp.expand_dims(self.mol.coords, jnp.arange(len(batch_dims))),
                    (*batch_dims, 1, 1),
                ),
                rs,
            ],
            axis=-2,
        )
        graph = self.graph_builder.build(positions)
        for apply_layer in self.apply_layers:
            graph = apply_layer(graph)
        return graph.nodes['elec']
