import jax.numpy as jnp
import pytest

from deepqmc.gnn.graph import GraphEdgeBuilder, MolecularGraphEdgeBuilder


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
        graph_edges = GraphEdgeBuilder(mask_self, offsets, mask_vals)(nodes, nodes)
        ndarrays_regression.check({'graph_edges': graph_edges})

    def test_molecular_graph_edge_builder(self, helpers, ndarrays_regression):
        mol = helpers.mol()
        hamil = helpers.hamil(mol)
        phys_conf = helpers.phys_conf()
        edge_types = ('ne', 'same', 'anti')
        graph_edges = MolecularGraphEdgeBuilder(
            hamil.n_nuc,
            hamil.n_up,
            hamil.n_down,
            edge_types,
            self_interaction=False,
        )(phys_conf)
        graph_edges = {k: edge.single_array for k, edge in graph_edges.items()}
        ndarrays_regression.check(graph_edges)


class TestGNN:
    def test_embedding(self, helpers, ndarrays_regression):
        mol = helpers.mol()
        hamil = helpers.hamil(mol)
        phys_conf = helpers.phys_conf(hamil)
        _gnn = helpers.init_conf('gnn')
        gnn = helpers.transform_model(_gnn, hamil, 8)
        params = helpers.init_model(gnn, phys_conf)
        emb = gnn.apply(params, phys_conf)
        ndarrays_regression.check(
            {'embedding': emb}, default_tolerance={'rtol': 1e-4, 'atol': 1e-6}
        )
