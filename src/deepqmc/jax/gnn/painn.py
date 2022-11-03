import haiku as hk
import jax.numpy as jnp
from jax import nn, ops

from ..hkext import MLP
from ..utils import norm
from .edge_features import PauliNetEdgeFeatures
from .gnn import GraphNeuralNetwork, MessagePassingLayer
from .graph import GraphNodes, difference_callback


class PaiNNLayer(MessagePassingLayer):
    def __init__(
        self,
        ilayer,
        shared,
        *,
        shared_g=False,
        n_layers_w=2,
        n_layers_h=2,
        n_layers_g=2,
        g_concat_norm=True,
        sv_connection=True,
        vs_connection=True,
        message_residual=True,
        subnet_kwargs=None,
        subnet_kwargs_by_lbl=None,
    ):
        super().__init__(ilayer, shared)
        assert not shared_g or self.has_vector_feat == self.edge_types
        self.shared_g = shared_g
        self.g_concat_norm = g_concat_norm
        self.sv_connection = sv_connection
        self.vs_connection = vs_connection
        self.message_residual = message_residual
        default_n_layers = {'w': n_layers_w, 'h': n_layers_h, 'g': n_layers_g}

        subnet_kwargs = subnet_kwargs or {}
        subnet_kwargs.setdefault('last_linear', True)
        subnet_kwargs.setdefault('activation', nn.silu)

        subnet_kwargs_by_lbl = subnet_kwargs_by_lbl or {}
        for lbl in self.subnet_labels:
            subnet_kwargs_by_lbl.setdefault(lbl, {})
            subnet_kwargs_by_lbl[lbl].setdefault('bias', lbl != 'w')
            for k, v in subnet_kwargs.items():
                subnet_kwargs_by_lbl[lbl].setdefault(k, v)
            subnet_kwargs_by_lbl[lbl].setdefault(
                'hidden_layers', ('log', default_n_layers[lbl])
            )
        out_size_fact = lambda typ: 3 if typ in self.has_vector_feat else 1
        self.w = {
            typ: MLP(
                self.edge_feat_dim[typ],
                out_size_fact(typ) * self.embedding_dim,
                name=f'w_{typ}',
                **subnet_kwargs_by_lbl['w'],
            )
            for typ in self.edge_types
        }

        self.h = {
            typ: hk.Embed(
                self.n_nuc if typ == 'ne' else 1 if self.n_up == self.n_down else 2,
                out_size_fact(typ) * self.embedding_dim,
                name=f'h_{typ}',
            )
            if self.first_layer
            else MLP(
                self.embedding_dim,
                out_size_fact(typ) * self.embedding_dim,
                name=f'h_{typ}',
                **subnet_kwargs_by_lbl['h'],
            )
            for typ in self.edge_types
        }
        g_input_size = (2 if g_concat_norm else 1) * self.embedding_dim
        self.g = (
            MLP(
                g_input_size,
                3 * self.embedding_dim,
                name='g',
                **subnet_kwargs_by_lbl['g'],
            )
            if shared_g
            else {
                typ: MLP(
                    g_input_size,
                    out_size_fact(typ) * self.embedding_dim,
                    name=f'g_{typ}',
                    **subnet_kwargs_by_lbl['g'],
                )
                for typ in self.edge_types
            }
        )
        for lbl in ['V', 'U']:
            setattr(
                self,
                lbl,
                {
                    typ: hk.Linear(
                        self.embedding_dim,
                        with_bias=False,
                        name=f'{lbl}_{typ}',
                        w_init=lambda shape, dtype: jnp.eye(shape[0], dtype=dtype),
                    )
                    for typ in self.has_vector_feat
                },
            )

    @classmethod
    @property
    def subnet_labels(cls):
        return {'w', 'h', 'g'}

    def get_update_edges_fn(self):
        return None

    def create_z(self, typ, phi_w, edge, n_node, send_node, recv_node):
        z = {'s': ops.segment_sum(phi_w['s'], edge.receivers, n_node)}
        if typ in self.has_vector_feat:
            v_update = phi_w['vv'][..., None] * send_node['v'][edge.senders]
            if self.vs_connection:
                v_update += (
                    phi_w['vs'][..., None] * edge.features['vector'][..., None, :]
                )
            z['v'] = ops.segment_sum(v_update, edge.receivers, n_node)
        if self.message_residual:
            z['s'] += recv_node['s']
            if typ in self.has_vector_feat:
                z['v'] += recv_node['v']
        return z

    def get_aggregate_edges_for_nodes_fn(self):
        def aggregate_edges_for_nodes_fn(nodes, edges):
            n_elec = nodes.electrons['s'].shape[-2]
            node_map = {
                'ne': (n_elec, nodes.nuclei, nodes.electrons, self.nuc_idxs),
                'same': (n_elec, nodes.electrons, nodes.electrons, self.spin_idxs),
                'anti': (n_elec, nodes.electrons, nodes.electrons, self.spin_idxs),
            }
            we = {
                typ: self.w[typ](edges[typ].features['scalars'])
                for typ in self.edge_types
            }
            hx = {
                typ: self.h[typ](
                    node_map[typ][3] if self.first_layer else node_map[typ][1]['s']
                )[edges[typ].senders]
                for typ in self.edge_types
            }
            phi_w = {
                typ: {
                    weh_lbl: weh
                    for weh_lbl, weh in zip(
                        ['s', 'vv', 'vs'], jnp.split(we * hx[typ], 3, axis=-1)
                    )
                }
                if typ in self.has_vector_feat
                else {'s': we * hx[typ]}
                for typ, we in we.items()
            }

            z = {
                typ: self.create_z(
                    typ,
                    phi_w,
                    edges[typ],
                    *node_map[typ][:3],
                )
                for typ, phi_w in phi_w.items()
            }
            return z

        return aggregate_edges_for_nodes_fn

    def combine_vectors(self, z):
        def linear_comb(W, v):
            lin_comb = jnp.swapaxes(W(jnp.swapaxes(v, -1, -2)), -1, -2)
            return lin_comb

        def lin_comb_all_typs(W, z):
            lin_comb_dict = {
                typ: linear_comb(W[typ], z[typ]['v']) for typ in self.has_vector_feat
            }
            return lin_comb_dict

        return lin_comb_all_typs(self.V, z), lin_comb_all_typs(self.U, z)

    def get_update_nodes_fn(self):
        def update_nodes_fn(nodes, z):
            Vv, Uv = self.combine_vectors(z)
            gs = {
                typ: self.g[typ](
                    jnp.concatenate(
                        [z_edge['s'], norm(Vv[typ], safe=self.safe)],
                        axis=-1,
                    )
                    if self.g_concat_norm and typ in self.has_vector_feat
                    else z_edge['s']
                )
                for typ, z_edge in z.items()
            }
            a = {
                typ: {
                    a_lbl: aa
                    for a_lbl, aa in zip(['ss', 'vv', 'sv'], jnp.split(g, 3, axis=-1))
                }
                if typ in self.has_vector_feat
                else {'ss': g}
                for typ, g in gs.items()
            }

            delta_s = sum(
                a[typ]['ss']
                + (
                    a[typ]['sv'] * jnp.einsum('pei,pei->pe', Uv[typ], Vv[typ])
                    if self.sv_connection and typ in self.has_vector_feat
                    else 0
                )
                for typ in self.edge_types
            )

            updated_s = jnp.tanh(delta_s)
            if self.has_vector_feat:
                delta_v = sum(
                    a[typ]['vv'][..., None] * Uv[typ] for typ in self.has_vector_feat
                )
                v_norm = norm(delta_v, safe=self.safe)
                updated_v = jnp.tanh(v_norm)[..., None] * delta_v / v_norm[..., None]
            else:
                updated_v = nodes.electrons['v']

            updated_nodes = nodes._replace(
                electrons={'s': updated_s, 'v': updated_v},
            )
            return updated_nodes

        return update_nodes_fn


class PaiNN(GraphNeuralNetwork):
    def __init__(
        self,
        mol,
        embedding_dim,
        *,
        cutoff=20.0,
        n_interactions=3,
        edge_feat_kwargs=None,
        edge_feat_kwargs_by_typ=None,
        concat_vectors=True,
        has_vector_feat=None,
        **gnn_kwargs,
    ):
        assert embedding_dim % 4 == 0
        _embedding_dim = embedding_dim // 4 if concat_vectors else embedding_dim
        n_nuc, n_up, n_down = mol.n_particles
        spin_idxs = jnp.array(
            (n_up + n_down) * [0] if n_up == n_down else n_up * [0] + n_down * [1]
        )
        nuc_idxs = jnp.arange(n_nuc)
        edge_feat_kwargs = edge_feat_kwargs or {}
        edge_feat_kwargs.setdefault('feature_dim', 8)
        edge_feat_kwargs.setdefault('cutoff', cutoff)
        edge_feat_kwargs.setdefault('difference', False)
        edge_feat_kwargs.setdefault('safe', True)
        edge_feat_kwargs_by_typ = edge_feat_kwargs_by_typ or {}
        nuc_idxs = jnp.arange(n_nuc)
        for typ in self.edge_types:
            edge_feat_kwargs_by_typ.setdefault(typ, {})
            for k, v in edge_feat_kwargs.items():
                edge_feat_kwargs_by_typ[typ].setdefault(k, v)
        share = {
            'edge_feat_dim': {
                typ: edge_feat_kwargs_by_typ[typ]['feature_dim']
                for typ in self.edge_types
            },
            'spin_idxs': spin_idxs,
            'nuc_idxs': nuc_idxs,
            'safe': edge_feat_kwargs['safe'],
            'has_vector_feat': set(has_vector_feat or self.edge_types),
        }
        super().__init__(
            mol,
            _embedding_dim,
            {typ: edge_feat_kwargs_by_typ[typ]['cutoff'] for typ in self.edge_types},
            n_interactions,
            **gnn_kwargs,
            share_with_layers=share,
        )
        self.edge_features = {
            typ: PauliNetEdgeFeatures(**kwargs)
            for typ, kwargs in edge_feat_kwargs_by_typ.items()
        }
        self.concat_vectors = concat_vectors

    def initial_embeddings(self):
        Y = hk.Embed(self.n_nuc, self.embedding_dim, name='NuclearEmbedding')
        X = hk.Embed(
            1 if self.n_up == self.n_down else 2,
            self.embedding_dim,
            name='ElectronicEmbedding',
        )

        v_nuc = jnp.zeros((self.n_nuc, self.embedding_dim, 3))
        v_elec = jnp.zeros((self.n_up + self.n_down, self.embedding_dim, 3))
        return GraphNodes(
            {'s': Y(self.nuc_idxs), 'v': v_nuc},
            {
                's': X(self.spin_idxs),
                'v': v_elec,
            },
        )

    @classmethod
    @property
    def edge_types(cls):
        return {'same', 'anti', 'ne'}

    def init_state(self, shape, dtype):
        zeros = jnp.zeros(shape, dtype)
        return {
            'ne': zeros,
            'same': (zeros, zeros),
            'anti': (zeros, zeros),
        }

    def edge_feature_callback(self, typ, *feature_callback_args):
        r = difference_callback(*feature_callback_args)
        scalars = self.edge_features[typ](r)
        vector = r / norm(r, safe=self.safe)[..., None]
        return {'scalars': scalars, 'vector': vector}

    @classmethod
    @property
    def layer_factory(cls):
        return PaiNNLayer

    def __call__(self, r):
        elec_embed = super().__call__(r)
        if self.concat_vectors:
            return jnp.concatenate(
                [
                    elec_embed['s'],
                    elec_embed['v'].reshape(self.n_up + self.n_down, -1),
                ],
                axis=-1,
            )
        else:
            return elec_embed['s']
