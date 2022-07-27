import haiku as hk
import jax.numpy as jnp
from jax import ops
from jax.tree_util import tree_map, tree_structure, tree_transpose

from deepqmc.jax.hkext import MLP

from .distbasis import DistanceBasis
from .graph import (
    DEFAULT_EDGE_KWARGS,
    Graph,
    GraphEdgesBuilder,
    GraphNodes,
    GraphUpdate,
)


class SchNetLayer(hk.Module):
    def __init__(
        self,
        ilayer,
        embedding_dim,
        kernel_dim,
        dist_feat_dim,
        distance_basis,
        shared_h=True,
        shared_g=False,
        w_subnet=None,
        h_subnet=None,
        g_subnet=None,
        *,
        n_layers_w=2,
        n_layers_h=1,
        n_layers_g=1,
    ):
        super().__init__(f'SchNetLayer_{ilayer}')

        def default_subnet_kwargs(n_layers):
            return {
                'hidden_layers': ('log', n_layers),
                'last_bias': False,
                'last_linear': True,
            }

        labels = ['same', 'anti', 'n']
        self.w = {
            lbl: MLP(
                dist_feat_dim,
                kernel_dim,
                name=f'w_{lbl}',
                **(w_subnet or default_subnet_kwargs(n_layers_w)),
            )
            for lbl in labels
        }
        self.h = (
            MLP(
                embedding_dim,
                kernel_dim,
                name='h',
                **(h_subnet or default_subnet_kwargs(n_layers_h)),
            )
            if shared_h
            else {
                lbl: MLP(
                    embedding_dim,
                    kernel_dim,
                    name=f'h_{lbl}',
                    **(h_subnet or default_subnet_kwargs(n_layers_h)),
                )
                for lbl in labels
            }
        )
        self.g = (
            MLP(
                kernel_dim,
                embedding_dim,
                name='g',
                **(g_subnet or default_subnet_kwargs(n_layers_g)),
            )
            if shared_g
            else {
                lbl: MLP(
                    kernel_dim,
                    embedding_dim,
                    name=f'g_{lbl}',
                    **(g_subnet or default_subnet_kwargs(n_layers_g)),
                )
                for lbl in labels
            }
        )
        self.distance_basis = distance_basis
        self.labels = labels
        self.shared_h = shared_h
        self.shared_g = shared_g

        self.forward = GraphUpdate(
            update_nodes_fn=self.get_update_nodes_fn(),
            update_edges_fn=self.get_update_edges_fn(),
            aggregate_edges_for_nodes_fn=self.get_aggregate_edges_for_nodes_fn(),
        )

    def __call__(self, graph):
        return self.forward(graph)

    def get_update_edges_fn(self):
        def update_edges_fn(nodes, edges):
            expanded = edges._replace(
                data=tree_map(lambda dists: self.distance_basis(dists), edges.data)
            )
            return expanded

        return update_edges_fn if self.distance_basis else None

    def get_aggregate_edges_for_nodes_fn(self):
        def aggregate_edges_for_nodes_fn(nodes, edges):
            n_elec = nodes.electrons.shape[-2]
            we_same, we_anti, we_n = (
                self.w[lbl](edges.data['distances'][lbl]) for lbl in self.labels
            )
            hx_same, hx_anti = (
                (self.h if self.shared_h else self.h[lbl])(
                    nodes.electrons[edges.senders[lbl]]
                )
                for lbl in self.labels[:2]
            )
            weh_same = we_same * hx_same
            weh_anti = we_anti * hx_anti
            weh_n = we_n * nodes.nuclei[edges.senders['n']]
            z_same = ops.segment_sum(
                data=weh_same, segment_ids=edges.receivers['same'], num_segments=n_elec
            )
            z_anti = ops.segment_sum(
                data=weh_anti, segment_ids=edges.receivers['anti'], num_segments=n_elec
            )
            z_n = ops.segment_sum(
                data=weh_n, segment_ids=edges.receivers['n'], num_segments=n_elec
            )
            return {
                'same': z_same,
                'anti': z_anti,
                'n': z_n,
            }

        return aggregate_edges_for_nodes_fn

    def get_update_nodes_fn(self):
        def update_nodes_fn(nodes, z):
            updated_nodes = nodes._replace(
                electrons=nodes.electrons
                + (
                    (self.g if self.shared_g else self.g['n'])(z['n'])
                    + (self.g if self.shared_g else self.g['same'])(z['same'])
                    + (self.g if self.shared_g else self.g['anti'])(z['anti'])
                )
            )
            return updated_nodes

        return update_nodes_fn


class SchNet(hk.Module):
    def __init__(
        self,
        n_nuc,
        n_up,
        n_down,
        coords,
        embedding_dim,
        dist_feat_dim=32,
        kernel_dim=128,
        n_interactions=3,
        cutoff=10.0,
        layer_kwargs=None,
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
        self.layers = [
            SchNetLayer(
                i,
                embedding_dim,
                kernel_dim,
                dist_feat_dim,
                DistanceBasis(dist_feat_dim, cutoff, envelope='nocusp')
                if i == 0
                else None,
                **(layer_kwargs or {}),
            )
            for i in range(n_interactions)
        ]

    def __call__(self, rs, graph_edges):
        def compute_distances(labels, positions):
            def dist(senders, receivers):
                return jnp.sqrt(((receivers - senders) ** 2).sum(axis=-1))

            data = {
                'distances': {
                    lbl: dist(
                        pos[0][graph_edges.senders[lbl]],
                        pos[1][graph_edges.receivers[lbl]],
                    )
                    for lbl, pos in zip(labels, positions)
                }
            }
            return data

        nuc_embedding = self.Y(self.nuclei_idxs)
        elec_embedding = self.X(self.spin_idxs)
        graph = Graph(
            GraphNodes(nuc_embedding, elec_embedding),
            graph_edges._replace(
                data=compute_distances(
                    ['n', 'same', 'anti'], [(self.coords, rs), (rs, rs), (rs, rs)]
                )
            ),
        )
        for layer in self.layers:
            graph = layer(graph)
        return graph.nodes.electrons


class SchNetEdgesBuilder:
    def __init__(self, mol, n_kwargs=None, same_kwargs=None, anti_kwargs=None):
        self.mol = mol
        n_nuc, self.n_up, self.n_down = mol.n_particles()
        n_elec = self.n_up + self.n_down
        get_kwargs = lambda kwargs: kwargs or DEFAULT_EDGE_KWARGS['SchNet']
        self.builders = {
            'n': GraphEdgesBuilder(
                **get_kwargs(n_kwargs),
                mask_self=False,
                send_mask_val=n_nuc,
                rec_mask_val=n_elec,
            ),
            'same': GraphEdgesBuilder(
                **get_kwargs(same_kwargs),
                mask_self=True,
                send_mask_val=n_elec,
                rec_mask_val=n_elec,
            ),
            'anti': GraphEdgesBuilder(
                **get_kwargs(anti_kwargs),
                mask_self=False,
                send_mask_val=n_elec,
                rec_mask_val=n_elec,
            ),
        }

    def __call__(self, rs):
        def transpose_cat(list_of_edges):
            def transpose_with_list(outer_structure, tree):
                return tree_transpose(tree_structure([0, 0]), outer_structure, tree)

            edges_of_lists = transpose_with_list(
                tree_structure(list_of_edges[0]), list_of_edges
            )
            edges = tree_map(
                lambda x: jnp.concatenate(x, axis=-1),
                edges_of_lists,
                is_leaf=lambda x: isinstance(x, list),
            )
            return edges

        batch_dims = rs.shape[:-2]
        coords = jnp.broadcast_to(
            jnp.expand_dims(self.mol.coords, jnp.arange(len(batch_dims))),
            (*batch_dims, *self.mol.coords.shape),
        )
        rs_up, rs_down = rs[..., : self.n_up, :], rs[..., self.n_up :, :]

        edges_same = [
            self.builders['same'](rs_up, rs_up),
            self.builders['same'](
                rs_down, rs_down, send_offset=self.n_up, rec_offset=self.n_up
            ),
        ]
        edges_same = transpose_cat(edges_same)

        edges_anti = [
            self.builders['anti'](rs_up, rs_down, rec_offset=self.n_up),
            self.builders['anti'](rs_down, rs_up, send_offset=self.n_up),
        ]
        edges_anti = transpose_cat(edges_anti)

        edges_n = self.builders['n'](coords, rs)

        edges = {'n': edges_n, 'same': edges_same, 'anti': edges_anti}
        return tree_transpose(
            tree_structure({'n': 0, 'same': 0, 'anti': 0}),
            tree_structure(edges_same),
            edges,
        )
