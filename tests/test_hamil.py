from functools import partial

import pytest

from deepqmc.hamil import MolecularHamiltonian


@pytest.mark.parametrize(
    'hamil,mol_kwargs',
    [
        (MolecularHamiltonian, {}),
        (MolecularHamiltonian, {'pp_type': 'ccECP'}),
    ],
    ids=['Molecular', 'Molecular+PP'],
)
class TestHamil:
    SAMPLE_SIZE = 5

    def test_init_sample(self, helpers, hamil, mol_kwargs, ndarrays_regression):
        hamil = helpers.hamil(helpers.mol(**mol_kwargs))
        phys_conf = helpers.phys_conf(hamil, n=self.SAMPLE_SIZE)
        ndarrays_regression.check({'rs': phys_conf.r})

    def test_local_energy(self, helpers, hamil, mol_kwargs, ndarrays_regression):
        hamil = helpers.hamil(helpers.mol(**mol_kwargs))
        phys_conf = helpers.phys_conf(hamil)
        wf, params = helpers.create_ansatz(hamil)
        E_loc, _ = hamil.local_energy(partial(wf.apply, params))(
            helpers.rng(), phys_conf
        )
        ndarrays_regression.check(
            {'E_loc': E_loc},
            default_tolerance={'rtol': 2e-4},
        )
