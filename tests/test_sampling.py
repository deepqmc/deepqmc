from functools import partial

import jax.numpy as jnp
import pytest

from deepqmc.sampling import (
    DecorrSampler,
    LangevinSampler,
    MetropolisSampler,
    MultiNuclearGeometrySampler,
    ResampledSampler,
    chain,
)
from deepqmc.sampling.nuclei_samplers import IdleNucleiSampler, no_elec_warp


@pytest.fixture(scope='class')
def wf(helpers, request):
    request.cls.mol = helpers.mol()
    hamil = helpers.hamil(request.cls.mol)
    ansatz, params = helpers.create_ansatz(hamil)
    request.cls.ansatz = ansatz
    request.cls.hamil = hamil
    request.cls.params = params


@pytest.mark.parametrize(
    'samplers',
    [
        (partial(MetropolisSampler, tau=0.1),),
        (partial(LangevinSampler, tau=0.1),),
        (DecorrSampler(length=20), partial(MetropolisSampler, tau=0.1, max_age=20)),
        (
            ResampledSampler(period=3),
            DecorrSampler(length=20),
            partial(MetropolisSampler, tau=0.1),
        ),
    ],
    ids=['Metropolis', 'Langevin', 'DecorrMetropolis', 'ResampledDecorrMetropolis'],
)
@pytest.mark.usefixtures('wf')
class TestSampling:
    SAMPLE_SIZE = 10

    def test_sampler_init(self, helpers, samplers, ndarrays_regression):
        sampler = chain(*samplers[:-1], samplers[-1](self.hamil, self.ansatz.apply))
        smpl_state = sampler.init(
            helpers.rng(), self.params, self.SAMPLE_SIZE, self.mol.coords
        )
        ndarrays_regression.check(
            helpers.flatten_pytree(smpl_state),
            default_tolerance={'rtol': 5e-4, 'atol': 1e-6},
        )

    def test_sampler_sample(self, helpers, samplers, ndarrays_regression):
        sampler = chain(*samplers[:-1], samplers[-1](self.hamil, self.ansatz.apply))
        smpl_state = sampler.init(
            helpers.rng(), self.params, self.SAMPLE_SIZE, self.mol.coords
        )
        for step in range(4):
            smpl_state, _, stats = sampler.sample(
                helpers.rng(step), smpl_state, self.params, self.mol.coords
            )
        ndarrays_regression.check(
            helpers.flatten_pytree({'smpl_state': smpl_state, 'stats': stats}),
        )


@pytest.mark.parametrize(
    'samplers',
    [
        (partial(MetropolisSampler, tau=0.1),),
        (partial(LangevinSampler, tau=0.1),),
    ],
    ids=['Metropolis', 'Langevin'],
)
@pytest.mark.usefixtures('wf')
class TestMultimoleculeSampling:
    SAMPLE_SIZE = 10
    N_CONFIG = 2

    def test_multi_nuclear_geometry_sampler_init(
        self, helpers, samplers, ndarrays_regression
    ):
        sampler = MultiNuclearGeometrySampler(
            chain(*samplers[:-1], samplers[-1](self.hamil, self.ansatz.apply)),
            IdleNucleiSampler(self.mol.charges),
            no_elec_warp,
            None,
            None,
        )
        smpl_state = sampler.init(
            helpers.rng(),
            self.params,
            self.SAMPLE_SIZE,
            jnp.stack([self.mol.coords for _ in range(self.N_CONFIG)]),
        )
        ndarrays_regression.check(
            helpers.flatten_pytree(smpl_state),
            default_tolerance={'rtol': 5e-4, 'atol': 1e-6},
        )

    def test_multi_nuclear_geometry_sampler_sample(
        self, helpers, samplers, ndarrays_regression
    ):
        sampler = MultiNuclearGeometrySampler(
            chain(*samplers[:-1], samplers[-1](self.hamil, self.ansatz.apply)),
            IdleNucleiSampler(self.mol.charges),
            no_elec_warp,
            None,
            None,
        )
        smpl_state = sampler.init(
            helpers.rng(),
            self.params,
            self.SAMPLE_SIZE,
            jnp.stack([self.mol.coords for _ in range(self.N_CONFIG)]),
        )
        mol_idxs = jnp.arange(2)
        for step in range(4):
            smpl_state, phys_conf, stats = sampler.sample(
                helpers.rng(step), smpl_state, self.params, mol_idxs
            )
        ndarrays_regression.check(
            helpers.flatten_pytree({'smpl_state': smpl_state, 'stats': stats}),
        )
