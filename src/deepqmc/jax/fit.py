import logging
from collections import namedtuple
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import kfac_jax
import optax

from .kfacext import GRAPH_PATTERNS
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
    params,
    opt,
    sample_wf,
    smpl_state,
    steps,
    *,
    clip_width,
    exclude_width=jnp.inf,
    clip_quantile=0.95,
    state_callback=None,
):
    vec_ansatz = jax.vmap(ansatz.apply, (None, 0, 0))

    @partial(jax.custom_jvp, nondiff_argnums=(1, 2))
    def loss_fn(params, state, rs):
        wf = lambda state, r: ansatz.apply(params, state, r)[0]
        E_loc, hamil_stats = jax.vmap(hamil.local_energy(wf))(state, rs)
        loss = jnp.mean(E_loc)
        stats = {
            'E_loc/mean': jnp.mean(E_loc),
            'E_loc/std': jnp.std(E_loc),
            'E_loc/max': jnp.max(E_loc),
            'E_loc/min': jnp.min(E_loc),
            **jax.tree_util.tree_map(jnp.mean, hamil_stats),
        }
        return loss, (E_loc, stats)

    @loss_fn.defjvp
    def loss_jvp(state, rs, primals, tangents):
        (params,) = primals
        loss, (E_loc, stats) = loss_fn(params, state, rs)
        E_loc_s, sigma = median_log_squeeze(E_loc, clip_width, clip_quantile)
        E_diff = E_loc_s - jnp.mean(E_loc_s)
        grad_ansatz = lambda params: vec_ansatz(params, state, rs)[0].log
        log_psi, log_psi_tangent = jax.jvp(grad_ansatz, primals, tangents)
        kfac_jax.register_normal_predictive_distribution(log_psi[:, None])
        loss_tangent = masked_mean(E_diff * log_psi_tangent, sigma < exclude_width)

        return (loss, stats), (loss_tangent, stats)

    energy_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    if isinstance(opt, optax.GradientTransformation):

        @jax.jit
        def train_step(rng, params, opt_state, smpl_state):
            rs, smpl_state, smpl_stats = sample_wf(rng, params, smpl_state)
            (loss, loss_stats), grads = energy_and_grad_fn(
                params, smpl_state['wf_state'], rs
            )
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            grad_norm = jax.tree_util.tree_reduce(
                lambda norm, x: norm + jnp.linalg.norm(x), grads, 0
            )
            stats = {
                'loss/value': loss,
                'loss/grad_norm': grad_norm,
                **smpl_stats,
                **loss_stats,
            }
            return params, opt_state, smpl_state, stats

        opt_state = opt.init(params)
    else:

        def train_step(rng, params, opt_state, smpl_state):
            rng_sample, rng_kfac = jax.random.split(rng)
            r, smpl_state = sample_wf(rng_sample, params, smpl_state)
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
        opt_state = opt.init(params, rng, smpl_state['r'])

    for step, rng in zip(steps, hk.PRNGSequence(rng)):
        new_params, new_opt_state, new_smpl_state, stats = train_step(
            rng, params, opt_state, smpl_state
        )
        if state_callback:
            state, overflow = state_callback(new_smpl_state['wf_state'])
            if overflow:
                smpl_state['wf_state'] = state
                new_params, new_opt_state, new_smpl_state, stats = train_step(
                    rng, params, opt_state, smpl_state
                )
        params, opt_state, smpl_state = new_params, new_opt_state, new_smpl_state
        yield step, TrainState(params, opt_state, smpl_state), {
            k: v.item() for k, v in stats.items()
        },
