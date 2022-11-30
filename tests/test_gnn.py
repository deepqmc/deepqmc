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


@pytest.fixture(scope='class')
def cutoff(request):
    request.cls.cutoff = 5.0


@pytest.mark.usefixtures('cutoff')
class TestGraph:
    @pytest.mark.parametrize(
        'mask_self,offsets,mask_vals,occupancy_len',
        [(True, (0, 0), (3, 3), 5), (False, (0, 3), (6, 6), 8)],
        ids=['mask_self=True', 'mask_self=False'],
    )
    def test_graph_edge_builder(
        self,
        helpers,
        nodes,
        mask_self,
        offsets,
        mask_vals,
        occupancy_len,
        ndarrays_regression,
    ):
        graph_edges, occupancy = GraphEdgeBuilder(
            self.cutoff, mask_self, offsets, mask_vals, difference_callback
        )(nodes, nodes, jnp.zeros(occupancy_len, dtype=int))
        ndarrays_regression.check(
            helpers.flatten_pytree({'graph_edges': graph_edges, 'occupancy': occupancy})
        )

    def test_molecular_graph_edge_builder(self, helpers, ndarrays_regression):
        mol = helpers.mol()
        rs = helpers.rs()
        edge_types = ('ne', 'same', 'anti')
        occupancy = {
            'ne': jnp.zeros(8),
            'same': (jnp.zeros(3), jnp.zeros(2)),
            'anti': (jnp.zeros(5), jnp.zeros(3)),
        }
        graph_edges, occupancy = MolecularGraphEdgeBuilder(
            *mol.n_particles,
            mol.coords,
            edge_types,
            {
                edge_type: {
                    'cutoff': self.cutoff,
                    'feature_callback': difference_callback,
                }
                for edge_type in edge_types
            },
        )(rs, occupancy)
        ndarrays_regression.check(
            helpers.flatten_pytree({'graph_edges': graph_edges, 'occupancy': occupancy})
        )


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
        ids=lambda x: ','.join(f'{k}={v}' for k, v in x.items())
        if isinstance(x, dict)
        else x,
    )
    def test_embedding(self, helpers, ndarrays_regression, kwargs):
        mol = helpers.mol()
        rs = helpers.rs()
        schnet = helpers.transform_model(SchNet, mol, 32, **kwargs)
        params, state = helpers.init_model(schnet, rs)
        emb, _ = schnet.apply(params, state, rs)
        ndarrays_regression.check({'embedding': emb})
