from functools import partial

import pytest
from jax import vmap

from deepqmc.sampling import (
    DecorrSampler,
    LangevinSampler,
    MetropolisSampler,
    ResampledSampler,
    chain,
)
from deepqmc.utils import check_overflow
from deepqmc.wf import PauliNet
from deepqmc.wf.base import state_callback


@pytest.fixture(scope='class')
def wf(helpers, request):
    hamil = helpers.hamil()
    paulinet = helpers.transform_model(PauliNet, hamil)
    params, state = vmap(paulinet.init, (None, 0), (None, 0))(
        helpers.rng(), helpers.rs(n=request.cls.SAMPLE_SIZE)
    )
    request.cls.wf = partial(paulinet.apply, params)
    request.cls.wf_state = state
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
    SAMPLE_SIZE = 100

    def test_sampler_init(self, helpers, samplers, ndarrays_regression):
        sampler = chain(*samplers[:-1], samplers[-1](self.hamil))
        smpl_state = sampler.init(
            helpers.rng(), self.wf, self.SAMPLE_SIZE, state_callback, self.wf_state
        )
        ndarrays_regression.check(
            helpers.flatten_pytree(smpl_state),
            default_tolerance={'rtol': 5e-4, 'atol': 1e-6},
        )

    def test_sampler_sample(self, helpers, samplers, ndarrays_regression):
        sampler = chain(*samplers[:-1], samplers[-1](self.hamil))
        smpl_state = sampler.init(
            helpers.rng(), self.wf, self.SAMPLE_SIZE, state_callback, self.wf_state
        )
        sample = check_overflow(state_callback, sampler.sample)
        for step in range(4):
            smpl_state, _, stats = sample(helpers.rng(step), smpl_state, self.wf)
        ndarrays_regression.check(
            helpers.flatten_pytree({'smpl_state': smpl_state, 'stats': stats}),
            default_tolerance={'rtol': 5e-4, 'atol': 1e-6},
        )
