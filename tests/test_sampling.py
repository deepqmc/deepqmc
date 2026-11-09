from functools import partial

import jax

import jax.numpy as jnp
import pytest

from deepqmc.sampling import (
    DecorrSampler,
    LangevinSampler,
    MetropolisSampler,
    MultiNuclearGeometrySampler,
)
from deepqmc.sampling.electron_sample_initializers import (
    AtomCenteredElectronInitializer,
    ShellBasedDistribution,
)
from deepqmc.sampling.sampling_utils import chain
from deepqmc.sampling.nuclei_samplers import (
    IdleNucleiSampler,
    PermutationNucleiSampler,
    no_elec_warp,
)


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
    ],
    ids=['Metropolis', 'Langevin', 'DecorrMetropolis'],
)
@pytest.mark.usefixtures('wf')
class TestSampling:
    SAMPLE_SIZE = 10

    def test_sampler_init(self, helpers, samplers, ndarrays_regression):
        sampler = chain(
            *samplers[:-1],
            samplers[-1](
                self.hamil,
                self.ansatz.apply,
                sample_initializer=AtomCenteredElectronInitializer(
                    atom_centered_distribution=ShellBasedDistribution()
                ),
            ),
        )
        smpl_state = sampler.init(
            helpers.rng(), self.params, self.SAMPLE_SIZE, self.mol.coords
        )
        ndarrays_regression.check(
            helpers.flatten_pytree(smpl_state),
            default_tolerance={'rtol': 5e-4, 'atol': 1e-6},
        )

    def test_sampler_sample(self, helpers, samplers, ndarrays_regression):
        sampler = chain(
            *samplers[:-1],
            samplers[-1](
                self.hamil,
                self.ansatz.apply,
                sample_initializer=AtomCenteredElectronInitializer(
                    atom_centered_distribution=ShellBasedDistribution()
                ),
            ),
        )
        smpl_state = sampler.init(
            helpers.rng(), self.params, self.SAMPLE_SIZE, self.mol.coords
        )
        for step in range(4):
            smpl_state, _, stats = sampler.sample(
                helpers.rng(step), smpl_state, self.params, self.mol.coords
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

    def test_multi_nuclear_geometry_sampler_init(
        self, helpers, samplers, ndarrays_regression
    ):
        sampler = MultiNuclearGeometrySampler(
            chain(
                *samplers[:-1],
                samplers[-1](
                    self.hamil,
                    self.ansatz.apply,
                    sample_initializer=AtomCenteredElectronInitializer(
                        atom_centered_distribution=ShellBasedDistribution()
                    ),
                ),
            ),
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
            chain(
                *samplers[:-1],
                samplers[-1](
                    self.hamil,
                    self.ansatz.apply,
                    sample_initializer=AtomCenteredElectronInitializer(
                        atom_centered_distribution=ShellBasedDistribution()
                    ),
                ),
            ),
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
            default_tolerance={'rtol': 5e-4, 'atol': 1e-6},
        )


class TestPermutationNucleiSampler:
    def test_nuclear_types_are_permuted_independently(self):
        # C2H2-like: C at 0, 1 and H at 2, 3
        sampler = PermutationNucleiSampler(jnp.array([6, 6, 1, 1]))
        perms = jax.vmap(sampler.permutation)(
            jax.random.split(jax.random.PRNGKey(0), 100)
        )
        swaps = {(bool(p[0] == 1), bool(p[2] == 3)) for p in perms}
        assert swaps == {(False, False), (False, True), (True, False), (True, True)}
        # only nuclei of the same type are exchanged
        assert jnp.all(jnp.sort(perms[:, :2], axis=1) == jnp.array([0, 1]))
        assert jnp.all(jnp.sort(perms[:, 2:], axis=1) == jnp.array([2, 3]))
