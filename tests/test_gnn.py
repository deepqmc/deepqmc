import jax.numpy as jnp
import pytest

from deepqmc.gnn import SchNet
from deepqmc.gnn.graph import (
    GraphEdgeBuilder,
    MolecularGraphEdgeBuilder,
    difference_callback,
)


@pytest.fixture
def nodes():
    return jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 6.0]])


class TestGraph:
    @pytest.mark.parametrize(
        'mask_self,offsets,mask_vals',
        [(True, (0, 0), (3, 3)), (False, (0, 3), (6, 6))],
        ids=['mask_self=True', 'mask_self=False'],
    )
    def test_graph_edge_builder(
        self,
        helpers,
        nodes,
        mask_self,
        offsets,
        mask_vals,
        ndarrays_regression,
    ):
        graph_edges = GraphEdgeBuilder(
            mask_self, offsets, mask_vals, difference_callback
        )(nodes, nodes)
        ndarrays_regression.check(helpers.flatten_pytree(graph_edges))

    def test_molecular_graph_edge_builder(self, helpers, ndarrays_regression):
        mol = helpers.mol()
        phys_conf = helpers.phys_conf()
        edge_types = ('ne', 'same', 'anti')
        graph_edges = MolecularGraphEdgeBuilder(
            *mol.n_particles,
            edge_types,
            {
                edge_type: {'feature_callback': difference_callback}
                for edge_type in edge_types
            },
        )(phys_conf)
        ndarrays_regression.check(helpers.flatten_pytree(graph_edges))


class TestSchNet:
    @pytest.mark.parametrize(
        'kwargs',
        [
            {},
            {'edge_feat_kwargs': {'difference': False}},
            {'layer_kwargs': {'deep_w': True}},
            {'layer_kwargs': {'residual': False}},
            {'layer_kwargs': {'shared_g': True}},
            {'layer_kwargs': {'shared_h': True}},
        ],
        ids=lambda x: (
            ','.join(f'{k}={v}' for k, v in x.items()) if isinstance(x, dict) else x
        ),
    )
    def test_embedding(self, helpers, ndarrays_regression, kwargs):
        mol = helpers.mol()
        phys_conf = helpers.phys_conf()
        schnet = helpers.transform_model(SchNet, mol, 32, **kwargs)
        params = helpers.init_model(schnet, phys_conf)
        emb = schnet.apply(params, phys_conf)
        ndarrays_regression.check(
            {'embedding': emb}, default_tolerance={'rtol': 1e-4, 'atol': 1e-6}
        )
