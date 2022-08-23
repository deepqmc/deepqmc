import logging
import operator
from collections import namedtuple
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import kfac_jax
import optax

from .kfacext import GRAPH_PATTERNS
from .utils import freeze_dict, masked_mean

__all__ = ()

log = logging.getLogger(__name__)
TrainState = namedtuple('TrainState', 'params state opt sampler')


def log_squeeze(x):
    sgn, x = jnp.sign(x), jnp.abs(x)
    return sgn * jnp.log1p((x + 1 / 2 * x ** 2 + x ** 3) / (1 + x ** 2))


def median_log_squeeze(x, width, quantile):
    x_median = jnp.median(x)
    x_diff = x - x_median
    quantile = jnp.quantile(jnp.abs(x_diff), quantile)
    width = width * quantile
    return (
        x_median + 2 * width * log_squeeze(x_diff / (2 * width)),
        jnp.abs(x_diff) / quantile,
    )


def fit_wf(
    rng,
    hamil,
    ansatz,
    opt,
    sampler,
    sample_size,
    steps,
    equilibration_steps=None,
    *,
    clip_width,
    exclude_width=jnp.inf,
    clip_quantile=0.95,
    state_callback=None,
):
    vec_ansatz = jax.vmap(ansatz.apply, (None, None, 0))

    @partial(jax.custom_jvp, nondiff_argnums=(1, 2))
    def loss_fn(params, state, rs):
        wf = lambda r: ansatz.apply(params, state, r)[0]
        E_loc = jax.vmap(hamil.local_energy(wf))(rs)
        loss = jnp.mean(E_loc)
        return loss, E_loc

    @loss_fn.defjvp
    def loss_jvp(state, rs, primals, tangents):
        (params,) = primals
        loss, E_loc = loss_fn(params, state, rs)
        E_loc_s, sigma = median_log_squeeze(E_loc, clip_width, clip_quantile)
        E_diff = E_loc_s - jnp.mean(E_loc_s)
        grad_ansatz = lambda params: vec_ansatz(params, state, rs)[0].log
        log_psi, log_psi_tangent = jax.jvp(grad_ansatz, primals, tangents)
        kfac_jax.register_normal_predictive_distribution(log_psi[:, None])
        loss_tangent = masked_mean(E_diff * log_psi_tangent, sigma < exclude_width)
        return (loss, E_loc), (loss_tangent, E_loc)

    energy_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @partial(jax.jit, static_argnums=(1,))
    def sample(params, state, smpl_state, rng):
        return sampler.sample(smpl_state, rng, partial(vec_ansatz, params, state))

    rng, rng_init_sample, rng_init_ansatz = jax.random.split(rng, 3)
    sample_for_init = hamil.init_sample(rng_init_sample, sample_size)
    params, state = ansatz.init(rng, sample_for_init[0])
    state = freeze_dict(state)
    num_params = jax.tree_util.tree_reduce(
        operator.add, jax.tree_map(lambda x: x.size, params)
    )
    log.info(f'Number of model parameters: {num_params}')

    if isinstance(opt, optax.GradientTransformation):

        @partial(jax.jit, static_argnums=(2,))
        def train_step(rng, params, state, opt_state, smpl_state):
            rs, smpl_state = sample(params, state, smpl_state, rng)
            (_, E_loc), grads = energy_and_grad_fn(params, state, rs)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, smpl_state['wf_state'], opt_state, smpl_state, E_loc

        opt_state = opt.init(params)
    else:

        def train_step(rng, params, opt_state, smpl_state):
            rng_sample, rng_kfac = jax.random.split(rng)
            r, smpl_state = sample(params, smpl_state, rng_sample)
            params, opt_state, stats = opt.step(
                params,
                opt_state,
                rng_kfac,
                batch=r,
                momentum=0,
                damping=1e-3,
            )
            return params, opt_state, smpl_state, stats['aux']

        opt = opt(
            value_and_grad_func=energy_and_grad_fn,
            l2_reg=0.0,
            value_func_has_aux=True,
            use_adaptive_learning_rate=True,
            auto_register_kwargs={'graph_patterns': GRAPH_PATTERNS},
        )
        opt_state = opt.init(params, rng, sample_for_init)

    smpl_state = sampler.init(rng, partial(vec_ansatz, params, state), sample_size)
    if state_callback:
        state, overflow = state_callback(state, smpl_state['wf_state'])
        if overflow:
            smpl_state = sampler.init(
                rng, partial(vec_ansatz, params, state), sample_size
            )
    if equilibration_steps:
        rng, rng_equilibrate = jax.random.split(rng, 2)
        for _, rng_eq in zip(equilibration_steps, hk.PRNGSequence(rng_equilibrate)):
            _, new_smpl_state = sample(params, state, smpl_state, rng_eq)
            if state_callback:
                new_state, overflow = state_callback(state, new_smpl_state['wf_state'])
                if overflow:
                    _, _, new_smpl_state = sample(params, new_state, smpl_state, rng_eq)
            smpl_state = new_smpl_state
            state = new_state
    train_state = params, state, opt_state, smpl_state
    for step, rng in zip(steps, hk.PRNGSequence(rng)):
        new_params, new_state, new_opt_state, new_smpl_state, E_loc = train_step(
            rng, *train_state
        )
        if state_callback:
            new_state, overflow = state_callback(state, new_state)
            if overflow:
                new_params, _, new_opt_state, new_smpl_state, E_loc = train_step(
                    rng, new_params, new_state, new_opt_state, new_smpl_state
                )
        (*train_state,) = new_params, new_state, new_opt_state, new_smpl_state
        yield step, TrainState(*train_state), E_loc
