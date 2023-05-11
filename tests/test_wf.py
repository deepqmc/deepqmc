import jax_dataclasses as jdc
from jax import grad

from deepqmc.physics import laplacian


class TestNeuralNetworkWaveFunction:
    def test_psi(self, helpers, ndarrays_regression):
        hamil = helpers.hamil()
        phys_conf = helpers.phys_conf(hamil)
        wf, params = helpers.create_ansatz(hamil)
        psi = wf.apply(params, phys_conf)
        ndarrays_regression.check(helpers.flatten_pytree(psi))

    def test_grad_psi(self, helpers, ndarrays_regression):
        hamil = helpers.hamil()
        phys_conf = helpers.phys_conf(hamil)
        wf, params = helpers.create_ansatz(hamil)
        grad_ansatz = grad(lambda params: wf.apply(params, phys_conf).log)
        grad_log_psi = grad_ansatz(params)
        ndarrays_regression.check(
            helpers.flatten_pytree(grad_log_psi),
            default_tolerance={'rtol': 1e-3, 'atol': 1e-5},
        )

    def test_laplace_psi(self, helpers, ndarrays_regression):
        hamil = helpers.hamil()
        phys_conf = helpers.phys_conf(hamil)
        wf, params = helpers.create_ansatz(hamil)
        lap_log_psis, quantum_force = laplacian(
            lambda r: wf.apply(params, jdc.replace(phys_conf, r=r.reshape(-1, 3))).log
        )(phys_conf.r.flatten())
        ndarrays_regression.check(
            {'lap_log_psis': lap_log_psis, 'quantum_force': quantum_force},
            default_tolerance={'rtol': 1e-3, 'atol': 1e-5},
        )
