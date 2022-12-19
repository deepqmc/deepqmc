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
    ids=lambda x: ','.join(f'{k}={v}' for k, v in x.items())
    if isinstance(x, dict)
    else x,
)
class TestPauliNet:
    def test_psi(self, helpers, kwargs, omni_kwargs, ndarrays_regression):
        params, state, paulinet, rs = helpers.create_paulinet(
            **kwargs, omni_kwargs=omni_kwargs
        )
        psi, _ = paulinet.apply(params, state, rs)
        ndarrays_regression.check(helpers.flatten_pytree(psi))

    def test_grad_psi(self, helpers, kwargs, omni_kwargs, ndarrays_regression):
        params, state, paulinet, rs = helpers.create_paulinet(
            **kwargs, omni_kwargs=omni_kwargs
        )
        grad_ansatz = grad(lambda params: paulinet.apply(params, state, rs)[0].log)
        grad_log_psi = grad_ansatz(params)
        ndarrays_regression.check(
            helpers.flatten_pytree(grad_log_psi),
            default_tolerance={'rtol': 2e-6, 'atol': 2e-6},
        )

    def test_laplace_psi(self, helpers, kwargs, omni_kwargs, ndarrays_regression):
        params, state, paulinet, rs = helpers.create_paulinet(
            **kwargs, omni_kwargs=omni_kwargs
        )
        lap_log_psis, quantum_force = laplacian(
            lambda r: paulinet.apply(params, state, r.reshape(-1, 3))[0].log
        )(rs.flatten())
        ndarrays_regression.check(
            {'lap_log_psis': lap_log_psis, 'quantum_force': quantum_force},
            default_tolerance={'rtol': 2e-6, 'atol': 2e-6},
        )
