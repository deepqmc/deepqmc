import logging
import operator
from collections import namedtuple
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import kfac_jax
import optax
from jax import lax

from .utils import masked_mean

__all__ = ()

log = logging.getLogger(__name__)
TrainState = namedtuple('TrainState', 'params opt sampler')


def log_squeeze(x):
    sgn, x = jnp.sign(x), jnp.abs(x)
    return sgn * jnp.log1p((x + 1 / 2 * x**2 + x**3) / (1 + x**2))


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
    edge_builder,
    sample_size,
    steps,
    equilibration_steps=None,
    *,
    clip_width,
    exclude_width=jnp.inf,
    clip_quantile=0.95,
):
    vec_ansatz = jax.jit(
        jax.vmap(ansatz.apply, (None, 0, 0) if edge_builder else (None, 0))
    )

    def loss_fn(params, r):
        wf = partial(ansatz.apply, params)
        E_loc = jax.vmap(hamil.local_energy(wf))(*r)
        psi = vec_ansatz(params, *r)
        kfac_jax.register_normal_predictive_distribution(psi.log[:, None])
        E_loc_s, sigma = median_log_squeeze(E_loc, clip_width, clip_quantile)
        loss = lax.stop_gradient(E_loc_s - E_loc_s.mean()) * psi.log
        loss = masked_mean(loss, sigma < exclude_width)
        return loss, E_loc

    def energy_and_grad_fn(params, r):
        grads, E_loc = jax.grad(loss_fn, has_aux=True)(params, r)
        return (jnp.mean(E_loc), E_loc), grads

    params = ansatz.init(
        rng, jnp.zeros(hamil.dim), edge_builder(hamil.init_sample(rng, 1)[0])
    )
    num_params = jax.tree_util.tree_reduce(
        operator.add, jax.tree_map(lambda x: x.size, params)
    )
    log.info(f'Number of model parameters: {num_params}')

    if isinstance(opt, optax.GradientTransformation):
        #  @jax.jit
        def sample(params, state, rng):
            return sampler.sample(state, rng, partial(vec_ansatz, params))

        @jax.jit
        def update_model(params, r, opt_state):
            (_, E_loc), grads = energy_and_grad_fn(params, r)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, E_loc

        def train_step(rng, params, opt_state, smpl_state):
            r, smpl_state = sample(params, smpl_state, rng)
            params, opt_state, E_loc = update_model(params, r, opt_state)
            return params, opt_state, smpl_state, E_loc

        opt_state = opt.init(params)
    else:
        #  @jax.jit
        def sample(params, state, rng):
            return sampler.sample(state, rng, partial(vec_ansatz, params))

        def train_step(rng, params, opt_state, smpl_state):
            r, smpl_state = sample(params, smpl_state, rng)
            params, opt_state, stats = opt.step(
                params, opt_state, None, batch=r, momentum=0, damping=1e-3
            )
            return params, opt_state, smpl_state, stats['aux']

        opt = opt(value_and_grad_func=energy_and_grad_fn, value_func_has_aux=True)
        opt_state = opt.init(params, rng, jnp.zeros((sample_size, *hamil.dim)))

    initial_ansatz = partial(vec_ansatz, params)
    smpl_state = sampler.init(rng, initial_ansatz, sample_size)
    if equilibration_steps:
        rng, rng_equilibrate = jax.random.split(rng, 2)
        for _, rng_eq in zip(equilibration_steps, hk.PRNGSequence(rng_equilibrate)):
            _, smpl_state = sampler.sample(smpl_state, rng_eq, initial_ansatz)
    train_state = params, opt_state, smpl_state
    for step, rng in zip(steps, hk.PRNGSequence(rng)):
        *train_state, E_loc = train_step(rng, *train_state)
        yield step, TrainState(*train_state), E_loc
