import logging
from collections import namedtuple

import haiku as hk
import jax
import jax.numpy as jnp
import kfac_jax
import optax

from .errors import NanError
from .ewm import ewm
from .kfacext import GRAPH_PATTERNS
from .utils import exp_normalize_mean, masked_mean

__all__ = ()

log = logging.getLogger(__name__)
TrainState = namedtuple('TrainState', 'params opt sampler')


def log_squeeze(x):
    sgn, x = jnp.sign(x), jnp.abs(x)
    return sgn * jnp.log1p((x + 1 / 2 * x**2 + x**3) / (1 + x**2))


def median_log_squeeze(x, width, quantile):
    x_median = jnp.nanmedian(x)
    x_diff = x - x_median
    quantile = jnp.nanquantile(jnp.abs(x_diff), quantile)
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
    log_dict=None,
    *,
    clip_width=1.0,
    exclude_width=jnp.inf,
    clip_quantile=0.95,
    state_callback=None,
):
    vec_ansatz = jax.vmap(ansatz.apply, (None, 0, 0))

    @jax.custom_jvp
    def loss_fn(params, state, batch):
        rs, weights = batch
        wf = lambda state, rs: ansatz.apply(params, state, rs)[0].log
        E_loc, hamil_stats = jax.vmap(hamil.local_energy(wf))(state, rs)
        loss = jnp.nanmean(E_loc * weights)
        stats = {
            'E_loc/mean': jnp.nanmean(E_loc),
            'E_loc/std': jnp.nanstd(E_loc),
            'E_loc/max': jnp.nanmax(E_loc),
            'E_loc/min': jnp.nanmin(E_loc),
            **jax.tree_util.tree_map(jnp.nanmean, hamil_stats),
        }
        return loss, (state, (E_loc, stats))

    @loss_fn.defjvp
    def loss_jvp(primals, tangents):
        rs, weights = primals[-1]
        loss, (_, (E_loc, stats)) = loss_fn(*primals)
        E_loc_s, sigma = median_log_squeeze(E_loc, clip_width, clip_quantile)
        E_diff = E_loc_s - jnp.mean(E_loc_s)
        grad_ansatz = lambda params: vec_ansatz(params, primals[1], rs)[0].log
        log_psi, log_psi_tangent = jax.jvp(grad_ansatz, primals[:1], tangents[:1])
        kfac_jax.register_normal_predictive_distribution(log_psi[:, None])
        loss_tangent = masked_mean(
            E_diff * log_psi_tangent * weights, sigma < exclude_width
        )

        return (loss, (primals[1], (E_loc, stats))), (
            loss_tangent,
            (primals[1], (E_loc, stats)),
        )

    energy_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    if isinstance(opt, optax.GradientTransformation):

        @jax.jit
        def train_step(rng, params, opt_state, smpl_state):
            rs, smpl_state, smpl_stats = sample_wf(rng, params, smpl_state)
            weights = exp_normalize_mean(smpl_state['log_weights'])
            (loss, (_, (E_loc, loss_stats))), grads = energy_and_grad_fn(
                params, smpl_state['wf_state'], (rs, weights)
            )
            updates, opt_state = opt.update(grads, opt_state, params)
            param_norm = jax.tree_util.tree_reduce(
                lambda norm, x: norm + jnp.linalg.norm(x), params, 0
            )
            update_norm = jax.tree_util.tree_reduce(
                lambda norm, x: norm + jnp.linalg.norm(x), updates, 0
            )

            params = optax.apply_updates(params, updates)
            grad_norm = jax.tree_util.tree_reduce(
                lambda norm, x: norm + jnp.linalg.norm(x), grads, 0
            )
            stats = {
                'opt/param_norm': param_norm,
                'opt/grad_norm': grad_norm,
                'opt/update_norm': update_norm,
                **smpl_stats,
                **loss_stats,
            }
            return TrainState(params, opt_state, smpl_state), stats, E_loc

        opt_state = opt.init(params)
    else:

        def train_step(rng, params, opt_state, smpl_state):
            rng_sample, rng_kfac = jax.random.split(rng)
            rs, smpl_state, smpl_stats = sample_wf(rng_sample, params, smpl_state)
            weights = exp_normalize_mean(jnp.copy(smpl_state['log_weights']))
            wf_state = jax.tree_util.tree_map(jnp.copy, smpl_state['wf_state'])
            params, opt_state, _, opt_stats = opt.step(
                params,
                opt_state,
                rng_kfac,
                func_state=wf_state,
                batch=(rs, weights),
                momentum=0,
            )
            stats = {
                'opt/param_norm': opt_stats['param_norm'],
                'opt/grad_norm': opt_stats['precon_grad_norm'],
                'opt/update_norm': opt_stats['update_norm'],
                **smpl_stats,
                **opt_stats['aux'][1],
            }
            return TrainState(params, opt_state, smpl_state), stats, opt_stats['aux'][0]

        opt = opt(
            value_and_grad_func=energy_and_grad_fn,
            l2_reg=0.0,
            value_func_has_aux=True,
            value_func_has_state=True,
            use_adaptive_learning_rate=False,
            auto_register_kwargs={'graph_patterns': GRAPH_PATTERNS},
            inverse_update_period=1,
            norm_constraint=3e-3,
            include_norms_in_stats=True,
            estimation_mode='fisher_exact',
            num_burnin_steps=0,
            damping_schedule=lambda s: 1e-3 / (1 + s / 6000),
            min_damping=1e-4,
            learning_rate_schedule=lambda s: 1e-1 / (1 + s / 6000),
        )
        opt_state = opt.init(
            params,
            rng,
            (smpl_state['r'], exp_normalize_mean(smpl_state['log_weights'])),
            func_state=smpl_state['wf_state'],
        )

    ewm_state = ewm()
    train_state = TrainState(params, opt_state, smpl_state)
    for step, rng in zip(steps, hk.PRNGSequence(rng)):
        new_train_state, train_stats, E_loc = train_step(rng, *train_state)
        if jnp.isnan(new_train_state[2]['psi'].log).any():
            raise NanError()
        if state_callback:
            state, overflow = state_callback(new_train_state[2]['wf_state'])
            if overflow:
                train_state[2]['wf_state'] = state
                new_train_state, stats, E_loc = train_step(rng, train_state)
        train_state = new_train_state
        ewm_state = ewm(train_stats['E_loc/mean'], ewm_state)
        train_stats = {
            'energy/ewm': ewm_state.mean,
            'energy/ewm_error': jnp.sqrt(ewm_state.sqerr),
            **train_stats,
        }
        if log_dict:
            log_dict['E_loc'] = E_loc
            log_dict['E_ewm'] = ewm_state.mean
            log_dict['sign_psi'] = train_state[2]['psi'].sign
            log_dict['log_psi'] = train_state[2]['psi'].log
        yield step, TrainState(params, opt_state, smpl_state), train_stats
