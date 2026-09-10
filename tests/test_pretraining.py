import importlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from deepqmc.types import PhysicalConfiguration

pretraining = importlib.import_module('deepqmc.pretrain.pretraining')


class ZeroTarget:
    def __init__(self, *args):
        pass

    def __call__(self, confs, coeffs, phys_conf):
        return jnp.zeros((1, 2, 2))


class StateCheckingSampler:
    def __init__(self, check_cache):
        self.check_cache = check_cache

    def sample(self, rng, state, params, mol_idxs):
        # Expose the input state through the public per-sample pretraining loss.
        value = (
            state['cached_w'] - params['w'][0] if self.check_cache else state['counter']
        )
        phys_conf = PhysicalConfiguration(
            R=jnp.zeros((1, 1, 2, 1, 3)),
            r=jnp.full((1, 1, 2, 2, 3), value),
            mol_idx=jnp.zeros((1, 1, 2), dtype=int),
        )
        return {**state, 'counter': state['counter'] + 1}, phys_conf, {}

    def update(self, state, params):
        return {**state, 'cached_w': params['w'][0]}


@pytest.mark.parametrize(
    'check_cache,merge_params,expected',
    [
        (False, False, [0, 1, 4]),
        (True, False, [1, 0.64, 0.4096]),
        (True, True, [1, 0.16, 0.0256]),
    ],
    ids=['propagate-state', 'refresh-cache', 'refresh-after-merge'],
)
def test_pretrain_sampler_state(monkeypatch, check_cache, merge_params, expected):
    monkeypatch.setattr(pretraining, 'PretrainTarget', ZeroTarget)
    if merge_params:
        # Ensure the cache uses the final merged parameters, not pre-merge ones.
        monkeypatch.setattr(
            pretraining,
            'pmap_merge_states',
            lambda params, keys: jax.tree.map(lambda x: x / 2, params),
        )

    def apply(params, phys_conf, return_mos):
        value = phys_conf.r[0, 0] + (params['w'] if check_cache else 0 * params['w'])
        return jnp.full((1, 1, 1), value), jnp.zeros((1, 1, 1))

    n_devices = jax.local_device_count()
    molecule_idx_sampler = SimpleNamespace(
        sample=lambda: jnp.zeros((n_devices, 1), dtype=int)
    )
    rows = list(
        pretraining.pretrain(
            jax.random.PRNGKey(0),
            SimpleNamespace(n_up=1, n_down=1),
            SimpleNamespace(apply=apply),
            {'w': jnp.ones((n_devices, 1))},
            optax.sgd(0.1),
            molecule_idx_sampler,
            StateCheckingSampler(check_cache),
            {'counter': jnp.zeros(n_devices), 'cached_w': jnp.ones(n_devices)},
            {
                'centers': None,
                'shells': None,
                'mo_coeffs': None,
                'confs': jnp.zeros((1, 1, 1, 2), dtype=int),
                'conf_coeffs': jnp.ones((1, 1, 1)),
            },
            ['w'] if merge_params else None,
            range(3),
        )
    )
    # With a fresh cache, SGD gives w -> 0.8*w; the optional merge halves it.
    np.testing.assert_allclose(
        [np.asarray(row[2]).mean() for row in rows], expected, rtol=1e-5
    )
