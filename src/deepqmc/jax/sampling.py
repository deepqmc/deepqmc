import logging
from functools import partial
from operator import add
from statistics import mean, stdev

import haiku as hk
import jax
import jax.numpy as jnp
from jax import lax

from .physics import pairwise_diffs, pairwise_self_distance
from .utils import multinomial_resampling, split_dict

__all__ = ()

log = logging.getLogger(__name__)


class MetropolisSampler:
    WALKER_STATE = ['r', 'psi', 'age']

    def __init__(self, hamil, *, tau, target_acceptance=0.57, max_age=None):
        self.hamil = hamil
        self.initial_tau = tau
        self.target_acceptance = target_acceptance
        self.max_age = max_age

    def _update(self, state, wf):
        psi, wf_state = jax.vmap(wf)(state['wf'], state['r'])
        state = {**state, 'psi': psi, 'wf': wf_state}
        return state

    def init(self, rng, wf, wf_state, n):
        state = {
            'r': self.hamil.init_sample(rng, n),
            'age': jnp.zeros(n, jnp.int32),
            'tau': jnp.array(self.initial_tau),
            'wf': wf_state,
        }
        state = self._update(state, wf)
        return state

    def _proposal(self, state, rng):
        r = state['r']
        return r + state['tau'] * jax.random.normal(rng, r.shape)

    def _acc_log_prob(self, state, prop):
        return 2 * (prop['psi'].log - state['psi'].log)

    def sample(self, state, rng, wf):
        rng_prop, rng_acc = jax.random.split(rng)
        prop = {
            'r': self._proposal(state, rng_prop),
            'age': jnp.zeros_like(state['age']),
            **{k: v for k, v in state.items() if k not in self.WALKER_STATE},
        }
        prop = self._update(prop, wf)
        log_prob = self._acc_log_prob(state, prop)
        accepted = log_prob > jnp.log(jax.random.uniform(rng_acc, log_prob.shape))
        if self.max_age:
            accepted = accepted | (state['age'] >= self.max_age)
        acceptance = accepted.astype(int).sum() / accepted.shape[0]
        if self.target_acceptance:
            prop['tau'] /= self.target_acceptance / jnp.max(
                jnp.stack([acceptance, jnp.array(0.05)])
            )
        state = {**state, 'age': state['age'] + 1}
        (prop, other), (state, _) = (
            split_dict(d, lambda k: k in self.WALKER_STATE) for d in (prop, state)
        )
        state = {
            **jax.tree_util.tree_map(
                lambda xp, x: jax.vmap(jnp.where)(accepted, xp, x), prop, state
            ),
            **other,
        }
        stats = {
            'sampling/acceptance': acceptance,
            'sampling/tau': state['tau'],
            'sampling/age/mean': jnp.mean(state['age']),
            'sampling/age/max': jnp.max(state['age']),
            'sampling/log_psi/mean': jnp.mean(state['psi'].log),
            'sampling/log_psi/std': jnp.std(state['psi'].log),
            'sampling/dists/mean': jnp.mean(pairwise_self_distance(state['r'])),
        }
        return state['r'], state, stats


class LangevinSampler(MetropolisSampler):
    WALKER_STATE = MetropolisSampler.WALKER_STATE + ['force']

    def _update(self, state, wf):
        @jax.vmap
        @partial(jax.value_and_grad, argnums=1, has_aux=True)
        def wf_and_force(state, r):
            psi, state = wf(state, r)
            return psi.log, (psi, state)

        (_, (psi, wf_state)), force = wf_and_force(state['wf'], state['r'])
        force = clean_force(force, state['r'], self.hamil.mol, tau=state['tau'])
        state = {**state, 'psi': psi, 'force': force, 'wf': wf_state}
        return state

    def _proposal(self, state, rng):
        r, tau = state['r'], state['tau']
        r = r + tau * state['force'] + jnp.sqrt(tau) * jax.random.normal(rng, r.shape)
        return r

    def _acc_log_prob(self, state, prop):
        log_G_ratios = jnp.sum(
            (state['force'] + prop['force'])
            * (
                (state['r'] - prop['r'])
                + state['tau'] / 2 * (state['force'] - prop['force'])
            ),
            axis=tuple(range(1, len(state['r'].shape))),
        )
        return log_G_ratios + 2 * (prop['psi'].log - state['psi'].log)


class DecorrSampler:
    def __init__(self, decorr):
        self.decorr = decorr

    def sample(self, state, rng, wf):
        sample = super().sample  # lax cannot parse super()
        state, stats = lax.scan(
            lambda state, rng: sample(state, rng, wf)[1:3],
            state,
            jax.random.split(rng, self.decorr),
        )
        stats = {k: v[-1] for k, v in stats.items()}
        return state['r'], state, stats


class ResampledSampler:
    def __init__(self, frequency):
        self.frequency = frequency

    def init(self, *args):
        state = super().init(*args)
        state = {
            **state,
            'step': jnp.array(0),
            'log_weight': jnp.zeros_like(state['psi']),
        }
        return state

    def sample(self, state, rng, wf):
        rng_re, rng_smpl = jax.random.split(rng)
        _, state, stats = super().sample(state, rng_smpl, wf)
        state['log_weight'] -= 2 * state['psi'].log
        state = self._update(state, wf)
        state['log_weight'] += 2 * state['psi'].log
        state['step'] += 1
        weight = jnp.exp(state['log_weight'])
        ess = jnp.sum(weight) ** 2 / jnp.sum(weight**2)
        stats['sampling/effective sample size'] = ess
        if state['step'] > self.frequency:
            idx = multinomial_resampling(rng_re, weight)
            state, other = split_dict(state, lambda k: k in self.WALKER_STATE)
            state = {
                **jax.tree_util.tree_map(lambda x: x[idx], state),
                **other,
                'log_weight': jnp.zeros_like(weight),
                'step': jnp.array(0),
            }
        return state['r'], state, stats


def chain(*samplers):
    name = 'Sampler'
    bases = tuple(map(type, samplers))
    for base in bases:
        name = name.replace('Sampler', base.__name__)
    chained = type(name, bases, {'__init__': lambda self: None})()
    for sampler in samplers:
        chained.__dict__.update(sampler.__dict__)
    return chained


def diffs_to_nearest_nuc(r, coords):
    z = pairwise_diffs(r, coords)
    idx = jnp.argmin(z[..., -1], axis=-1)
    return z[jnp.arange(len(r)), idx], idx


def crossover_parameter(z, f, charge):
    z, z2 = z[..., :3], z[..., 3]
    eps = jnp.finfo(f.dtype).eps
    z_unit = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
    f_unit = f / jnp.clip(jnp.linalg.norm(f, axis=-1, keepdims=True), eps, None)
    Z2z2 = charge**2 * z2
    return (1 + jnp.sum(f_unit * z_unit, axis=-1)) / 2 + Z2z2 / (10 * (4 + Z2z2))


def clean_force(force, r, mol, *, tau):
    z, idx = diffs_to_nearest_nuc(jnp.reshape(r, (-1, 3)), mol.coords)
    a = crossover_parameter(z, jnp.reshape(force, (-1, 3)), mol.charges[idx])
    z, a = jnp.reshape(z, (len(r), -1, 4)), jnp.reshape(a, (len(r), -1))
    av2tau = a * jnp.sum(force**2, axis=-1) * tau
    # av2tau can be small or zero, so the following expression must handle that
    factor = 2 / (jnp.sqrt(1 + 2 * av2tau) + 1)
    force = factor[..., None] * force
    eps = jnp.finfo(r.dtype).eps
    norm_factor = jnp.minimum(
        1.0,
        jnp.sqrt(z[..., -1])
        / (tau * jnp.clip(jnp.linalg.norm(force, axis=-1), eps, None)),
    )
    force = force * norm_factor[..., None]
    return force


def equilibrate(
    rng,
    wf,
    sampler,
    state,
    criterion,
    steps,
    state_callback=None,
    *,
    block_size,
    n_blocks=5,
):
    @jax.jit
    def sample_wf(state, rng):
        return sampler.sample(state, rng, wf)

    buffer_size = block_size * n_blocks
    buffer = []
    for step, rng in zip(steps, hk.PRNGSequence(rng)):
        r, state, stats = sample_wf(state_prev := state, rng)
        if state_callback:
            wf_state, overflow = state_callback(state['wf'])
            if overflow:
                state = state_prev
                state['wf'] = wf_state
                continue
        yield step, state, stats
        buffer = [*buffer[-buffer_size + 1 :], criterion(r)]
        if len(buffer) < buffer_size:
            continue
        b1, b2 = buffer[:block_size], buffer[-block_size:]
        if abs(mean(b1) - mean(b2)) < min(stdev(b1), stdev(b2)):
            break


def init_sampling(
    rng,
    hamil,
    ansatz,
    state_callback,
    *,
    params=None,
    sampler=MetropolisSampler,
    sampler_kwargs=None,
):
    sampler = sampler(hamil, **(sampler_kwargs or {}))

    rng_hamil, rng_ansatz, rng_smpl = jax.random.split(rng, 3)
    init_smpl = hamil.init_sample(rng_hamil, sampler.sample_size)
    maybe_params, wf_state = jax.vmap(ansatz.init, (None, 0), (None, 0))(
        rng_ansatz, init_smpl
    )
    params = params or maybe_params
    n_params = jax.tree_util.tree_reduce(
        add, jax.tree_util.tree_map(lambda x: x.size, params)
    )
    log.info(f'Number of WF parameters: {n_params}')

    wf = partial(ansatz.apply, params)
    smpl_state = sampler.init(rng_smpl, wf, wf_state)
    if state_callback:
        wf_state, overflow = state_callback(smpl_state['wf_state'])
        if overflow:
            smpl_state = sampler.init(rng_smpl, wf, wf_state)

    return params, smpl_state, sampler
