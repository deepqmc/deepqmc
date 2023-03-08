import jax_dataclasses as jdc
import pytest
from jax import grad

from deepqmc.physics import laplacian


@pytest.mark.parametrize(
    'kwargs,omni_kwargs',
    [
        ({}, {}),
        ({}, {'jastrow': False}),
        ({'backflow_channels': 2}, {}),
        ({'confs': [[0, 1, 0, 1], [0, 2, 0, 2]]}, {}),
    ],
    ids=lambda x: (
        ','.join(f'{k}={v}' for k, v in x.items()) if isinstance(x, dict) else x
    ),
)
class TestPauliNet:
    def test_psi(self, helpers, kwargs, omni_kwargs, ndarrays_regression):
        params, state, paulinet, phys_conf = helpers.create_paulinet(
            paulinet_kwargs={**kwargs, 'omni_kwargs': omni_kwargs}
        )
        psi, _ = paulinet.apply(params, state, phys_conf)
        ndarrays_regression.check(helpers.flatten_pytree(psi))

    def test_grad_psi(self, helpers, kwargs, omni_kwargs, ndarrays_regression):
        params, state, paulinet, phys_conf = helpers.create_paulinet(
            paulinet_kwargs={**kwargs, 'omni_kwargs': omni_kwargs}
        )
        grad_ansatz = grad(
            lambda params: paulinet.apply(params, state, phys_conf)[0].log
        )
        grad_log_psi = grad_ansatz(params)
        ndarrays_regression.check(
            helpers.flatten_pytree(grad_log_psi),
            default_tolerance={'rtol': 1e-3, 'atol': 1e-5},
        )

    def test_laplace_psi(self, helpers, kwargs, omni_kwargs, ndarrays_regression):
        params, state, paulinet, phys_conf = helpers.create_paulinet(
            paulinet_kwargs={**kwargs, 'omni_kwargs': omni_kwargs}
        )
        lap_log_psis, quantum_force = laplacian(
            lambda r: paulinet.apply(
                params, state, jdc.replace(phys_conf, r=r.reshape(-1, 3))
            )[0].log
        )(phys_conf.r.flatten())
        ndarrays_regression.check(
            {'lap_log_psis': lap_log_psis, 'quantum_force': quantum_force},
            default_tolerance={'rtol': 1e-3, 'atol': 1e-5},
        )
