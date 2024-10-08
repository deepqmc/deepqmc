import pytest

from deepqmc.hamil import MolecularHamiltonian


@pytest.mark.parametrize(
    'hamil,ecp_type',
    [
        (MolecularHamiltonian, None),
        (MolecularHamiltonian, 'ccECP'),
    ],
    ids=['Molecular', 'Molecular+PP'],
)
class TestHamil:
    SAMPLE_SIZE = 5

    def test_init(self, helpers, hamil, ecp_type, ndarrays_regression):
        hamil = helpers.hamil(helpers.mol(), ecp_type=ecp_type)
        ndarrays_regression.check(
            {
                'n_up': hamil.n_up,
                'n_down': hamil.n_down,
                'ns_valence': hamil.ns_valence,
                'pp_mask': hamil.ecp_mask,
            }
        )

    def test_init_sample(self, helpers, hamil, ecp_type, ndarrays_regression):
        hamil = helpers.hamil(helpers.mol(), ecp_type=ecp_type)
        phys_conf = helpers.phys_conf(hamil, n=self.SAMPLE_SIZE)
        ndarrays_regression.check({'rs': phys_conf.r})

    def test_local_energy(self, helpers, hamil, ecp_type, ndarrays_regression):
        hamil = helpers.hamil(helpers.mol(), ecp_type=ecp_type)
        phys_conf = helpers.phys_conf(hamil)
        ansatz, params = helpers.create_ansatz(hamil)
        E_loc, _ = hamil.local_energy(ansatz.apply)(helpers.rng(), params, phys_conf)
        ndarrays_regression.check(
            {'E_loc': E_loc},
            default_tolerance={'rtol': 2e-4},
        )
