import jax.numpy as jnp
import pytest
from conftest import Helpers

from deepqmc.hamil import MolecularHamiltonian
from deepqmc.hamil.qho import QHOHamiltonian
from deepqmc.wf import PauliNet
from deepqmc.wf.qho import QHOAnsatz

dim = 10


@pytest.mark.parametrize(
    'hamil,hamil_kwargs,wf',
    [
        (MolecularHamiltonian, {'mol': Helpers.mol()}, PauliNet),
        (
            QHOHamiltonian,
            {'dim': dim, 'mass': jnp.ones(dim), 'nu': jnp.ones(dim)},
            QHOAnsatz,
        ),
    ],
    ids=['Molecular', 'QHO'],
)
class TestHamil:
    SAMPLE_SIZE = 5

    def test_init_sample(self, helpers, hamil, hamil_kwargs, wf, ndarrays_regression):
        hamil = hamil(**hamil_kwargs)
        phys_conf = hamil.init_sample(helpers.rng(), Helpers.R(), self.SAMPLE_SIZE)
        ndarrays_regression.check({'rs': phys_conf.r})

    def test_local_energy(self, helpers, hamil, hamil_kwargs, wf, ndarrays_regression):
        hamil = hamil(**hamil_kwargs)
        phys_conf = hamil.init_sample(helpers.rng(), Helpers.R(), self.SAMPLE_SIZE)[0]
        wf = helpers.transform_model(wf, hamil)
        params, state = helpers.init_model(wf, phys_conf)
        E_loc, *_ = hamil.local_energy(
            lambda state, phys_conf: wf.apply(params, state, phys_conf)[0]
        )(helpers.rng(), state, phys_conf)
        ndarrays_regression.check({'E_loc': E_loc})
