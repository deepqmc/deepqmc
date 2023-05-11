from functools import partial

import pytest

from deepqmc.sampling import (
    DecorrSampler,
    LangevinSampler,
    MetropolisSampler,
    MultimoleculeSampler,
    ResampledSampler,
    chain,
)


@pytest.fixture(scope='class')
def wf(helpers, request):
    request.cls.mol = helpers.mol()
    hamil = helpers.hamil(request.cls.mol)
    wf, params = helpers.create_ansatz(hamil)
    request.cls.wf = partial(wf.apply, params)
    request.cls.hamil = hamil


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
        sampler = chain(*samplers[:-1], samplers[-1](self.hamil))
        smpl_state = sampler.init(
            helpers.rng(), self.wf, self.SAMPLE_SIZE, self.mol.coords
        )
        ndarrays_regression.check(
            helpers.flatten_pytree(smpl_state),
            default_tolerance={'rtol': 5e-4, 'atol': 1e-6},
        )

    def test_sampler_sample(self, helpers, samplers, ndarrays_regression):
        sampler = chain(*samplers[:-1], samplers[-1](self.hamil))
        smpl_state = sampler.init(
            helpers.rng(), self.wf, self.SAMPLE_SIZE, self.mol.coords
        )
        for step in range(4):
            smpl_state, _, stats = sampler.sample(
                helpers.rng(step), smpl_state, self.wf, self.mol.coords
            )
        ndarrays_regression.check(
            helpers.flatten_pytree({'smpl_state': smpl_state, 'stats': stats}),
            default_tolerance={'rtol': 5e-4, 'atol': 1e-6},
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

    def test_multimolecule_sampler_init(self, helpers, samplers, ndarrays_regression):
        sampler = MultimoleculeSampler(
            chain(*samplers[:-1], samplers[-1](self.hamil)),
            [self.mol for _ in range(self.N_CONFIG)],
        )
        smpl_state = sampler.init(helpers.rng(), self.wf, self.SAMPLE_SIZE)
        ndarrays_regression.check(
            helpers.flatten_pytree(smpl_state),
            default_tolerance={'rtol': 5e-4, 'atol': 1e-6},
        )

    def test_multimolecule_sampler_sample(self, helpers, samplers, ndarrays_regression):
        sampler = MultimoleculeSampler(
            chain(*samplers[:-1], samplers[-1](self.hamil)),
            [self.mol for _ in range(self.N_CONFIG)],
        )
        smpl_state = sampler.init(helpers.rng(), self.wf, self.SAMPLE_SIZE)
        select_idxs = sampler.select_idxs(self.SAMPLE_SIZE, 0)
        for step in range(4):
            smpl_state, phys_conf, stats = sampler.sample(
                helpers.rng(step), smpl_state, self.wf, select_idxs
            )
        ndarrays_regression.check(
            helpers.flatten_pytree({'smpl_state': smpl_state, 'stats': stats}),
            default_tolerance={'rtol': 5e-4, 'atol': 1e-6},
        )
