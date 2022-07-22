from functools import partial

import haiku as hk
import jax.numpy as jnp
from jax import jit, ops, random

from deepqmc.jax.jaxext import SSP
from deepqmc.jax.wf.paulinet.distbasis import DistanceBasis
from deepqmc.jax.wf.paulinet.graph import GraphBuilder, GraphNetwork


def subnet(rng, layer_dims):
    def _subnet_forward(x):
        subnet = hk.nets.MLP(layer_dims[1:], activation=SSP)
        return subnet(x)

    _subnet = hk.without_apply_rng(hk.transform(_subnet_forward))
    params = _subnet.init(rng, jnp.zeros((1, layer_dims[0])))
    return _subnet.apply, params


class SchNetLayer:
    def __init__(
        self,
        rng,
        embedding_dim,
        kernel_dim,
        dist_feat_dim,
        shared_h=True,
        shared_g=False,
    ):
        self.kernel_dim = kernel_dim
        labels = ['same', 'anti', 'n']
        self.labels = labels
        self.shared_h = shared_h
        self.shared_g = shared_g
        rng, *rng_ws = random.split(rng, len(labels) + 1)
        rng, *rng_hs = random.split(rng, 2 if shared_h else len(labels) + 1)
        rng, *rng_gs = random.split(rng, 2 if shared_g else len(labels) + 1)
        subnet_params = {}
        apply_subnet = {}

        def create_subnet(rngs, labels, dims):
            if labels is None:
                return subnet(rngs[0], dims)
            else:
                applies, params = {}, {}
                for lbl, rng in zip(labels, rngs):
                    applies[lbl], params[lbl] = subnet(rng, dims)
                return applies, params

        apply_subnet['w'], subnet_params['w'] = create_subnet(
            rng_ws, labels, [dist_feat_dim, dist_feat_dim, kernel_dim]
        )
        apply_subnet['h'], subnet_params['h'] = create_subnet(
            rng_hs, None if shared_h else labels, [embedding_dim, kernel_dim]
        )
        apply_subnet['g'], subnet_params['g'] = create_subnet(
            rng_gs, None if shared_g else labels, [kernel_dim, embedding_dim]
        )
        self.apply_subnet = apply_subnet
        self.subnet_params = subnet_params

    def get_aggregate_edges_for_nodes_fn(self):
        apply_w_same, apply_w_anti, apply_w_n = [
            self.apply_subnet['w'][lbl] for lbl in self.labels
        ]
        apply_h = self.apply_subnet['h'] if self.shared_h else None

        @partial(jit, static_argnames='n_nodes')
        def aggregate_edges_for_nodes_fn(
            subnet_params, nodes, edges, senders, receivers, n_nodes
        ):
            dist, e_typ = edges['dist'], edges['type'][:, None]
            zeros = jnp.zeros_like(
                dist, shape=(*dist.shape[:-1], nodes['nuc'].shape[-1])
            )
            nuc_or_zero = jnp.where(
                e_typ == 1, apply_w_n(subnet_params['w']['n'], dist), zeros
            )
            anti_or_nuc = jnp.where(
                e_typ == 4, apply_w_anti(subnet_params['w']['anti'], dist), nuc_or_zero
            )
            we = jnp.where(
                e_typ == 3, apply_w_same(subnet_params['w']['same'], dist), anti_or_nuc
            )
            hx = jnp.concatenate(
                [nodes['nuc'], apply_h(subnet_params['h'], nodes['elec'])], axis=-2
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
        apply_g_same, apply_g_anti, apply_g_n = [
            self.apply_subnet['g'][lbl] for lbl in self.labels
        ]

        @jit
        def update_node_fn(subnet_params, nodes, z):
            def eval_g(apply_g, params, message):
                n_nuclei = nodes['nuc'].shape[-2]
                n_particles = nodes['elec'].shape[-2] + n_nuclei
                return apply_g(
                    params,
                    message.reshape(-1, n_particles, self.kernel_dim)[:, n_nuclei:],
                )

            nodes['elec'] += (
                eval_g(apply_g_n, subnet_params['g']['n'], z['nuc'])
                + eval_g(apply_g_same, subnet_params['g']['same'], z['same'])
                + eval_g(apply_g_anti, subnet_params['g']['anti'], z['anti'])
            )
            return nodes

        return update_node_fn


class SchNet(hk.Module):
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
        super().__init__()
        n_nuc, n_up, n_down = mol.n_particles()
        self.mol = mol

        self.X = hk.Embed(1 if n_up == n_down else 2, embedding_dim)
        self.Y = hk.Embed(n_nuc, kernel_dim)
        self.spin_idxs = jnp.array(
            (n_up + n_down) * [0] if n_up == n_down else n_up * [0] + n_down * [1]
        )
        self.nuclei_idxs = jnp.arange(n_nuc)
        self.graph_builder = GraphBuilder(
            n_nuc,
            n_up,
            n_down,
            cutoff,
            DistanceBasis(dist_feat_dim, envelope='nocusp'),
        )
        *rng_layers = random.split(rng, n_interactions)
        self.layers = [
            SchNetLayer(rng_layer, embedding_dim, kernel_dim, dist_feat_dim)
            for rng_layer in rng_layers
        ]
        self.apply_layers = [
            GraphNetwork(
                update_node_fn=partial(layer.get_update_node_fn(), layer.subnet_params),
                aggregate_edges_for_nodes_fn=partial(
                    layer.get_aggregate_edges_for_nodes_fn(), layer.subnet_params
                ),
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
        nuc_embedding = self.Y(self.nuclei_idxs)
        elec_embedding = self.X(self.spin_idxs)
        graph = self.graph_builder.build(positions, nuc_embedding, elec_embedding)
        for apply_layer in self.apply_layers:
            graph = apply_layer(graph)
        return graph.nodes['elec']
