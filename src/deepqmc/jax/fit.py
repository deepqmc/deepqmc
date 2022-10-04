import copy
import logging
from collections import namedtuple
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import kfac_jax
import optax

from .errors import NanError
from .ewm import ewm
from .kfacext import GRAPH_PATTERNS
from .utils import exp_normalize_mean, masked_mean, tree_norm

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


def fit_wf(  # noqa: C901
    rng,
    hamil,
    ansatz,
    state_callback,
    params,
    opt,
    sample_wf,
    smpl_state,
    steps,
    log_dict=None,
    rewind=0,
    *,
    clip_width=1.0,
    exclude_width=jnp.inf,
    clip_quantile=0.95,
):
    @partial(jax.custom_jvp, nondiff_argnums=(1, 2))
    def loss_fn(params, state, batch):
        r, weight = batch
        wf = lambda state, r: ansatz.apply(params, state, r)[0]
        E_loc, hamil_stats = jax.vmap(hamil.local_energy(wf))(state, r)
        loss = jnp.nanmean(E_loc * weight)
        stats = {
            'E_loc/mean': jnp.nanmean(E_loc),
            'E_loc/std': jnp.nanstd(E_loc),
            'E_loc/max': jnp.nanmax(E_loc),
            'E_loc/min': jnp.nanmin(E_loc),
            **jax.tree_util.tree_map(jnp.nanmean, hamil_stats),
        }
        return loss, (None, (E_loc, stats))
        # - kfac-jax docs says the API should be (loss, state, aux), but that's
        #   wrong, it in fact expects (loss, (state, aux)):
        #   https://github.com/deepmind/kfac-jax/blob/17831f5a0621b0259c644503556ee7f65acdf0c5/kfac_jax/_src/optimizer.py#L1380-L1383
        # - we're passing out None as state to satisfy KFAC API, but we don't
        #   actually use it (hence None)

    @loss_fn.defjvp
    def loss_jvp(state, batch, primals, tangents):
        r, weight = batch
        (params,) = primals
        loss, other = loss_fn(params, state, batch)
        # other is (state, aux) as per kfac-jax's convention
        _, (E_loc, _) = other
        E_loc_s, sigma = median_log_squeeze(E_loc, clip_width, clip_quantile)

        def log_likelihood(params):  # log(psi(theta))
            wf = lambda state, r: ansatz.apply(params, state, r)[0]
            return jax.vmap(wf)(state, r).log

        log_psi, log_psi_tangent = jax.jvp(log_likelihood, primals, tangents)
        kfac_jax.register_normal_predictive_distribution(log_psi[:, None])
        loss_tangent = (E_loc_s - jnp.mean(E_loc_s)) * log_psi_tangent * weight
        loss_tangent = masked_mean(loss_tangent, sigma < exclude_width)
        return (loss, other), (loss_tangent, other)
        # jax.custom_jvp has actually no official support for auxiliary output.
        # the second other in the tangent output should be in fact other_tangent.
        # we just output the same thing to satisfy jax's API requirement with
        # the understanding that we'll never need other_tangent

    energy_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    if isinstance(opt, optax.GradientTransformation):

        @jax.jit
        def train_step(rng, params, opt_state, smpl_state):
            r, smpl_state, smpl_stats = sample_wf(rng, params, smpl_state)
            weight = exp_normalize_mean(smpl_state['log_weights'])
            (loss, (_, (E_loc, loss_stats))), grads = energy_and_grad_fn(
                params, smpl_state['wf_state'], (r, weight)
            )
            updates, opt_state = opt.update(grads, opt_state, params)
            param_norm, update_norm, grad_norm = map(
                tree_norm, [params, updates, grads]
            )
            params = optax.apply_updates(params, updates)
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
            r, smpl_state, smpl_stats = sample_wf(rng_sample, params, smpl_state)
            weight = exp_normalize_mean(jnp.copy(smpl_state['log_weights']))
            wf_state = jax.tree_util.tree_map(jnp.copy, smpl_state['wf_state'])
            params, opt_state, _, opt_stats = opt.step(
                params,
                opt_state,
                rng_kfac,
                func_state=wf_state,
                batch=(r, weight),
                momentum=0,
                damping=5e-4,
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
            norm_constraint=1e-3,
            include_norms_in_stats=True,
            estimation_mode='fisher_exact',
            num_burnin_steps=0,
            learning_rate_schedule=lambda s: 1e-2 / (1 + s / 6000),
        )
        opt_state = opt.init(
            params,
            rng,
            (smpl_state['r'], exp_normalize_mean(smpl_state['log_weights'])),
            func_state=smpl_state['wf_state'],
        )

    history = []
    ewm_state = ewm()
    train_state = TrainState(params, opt_state, smpl_state)
    for step, rng in zip(steps, hk.PRNGSequence(rng)):
        history.append(copy.deepcopy(train_state))
        history = history[-max(rewind, 1) :]
        done = False
        while not done:
            new_train_state, train_stats, E_loc = train_step(rng, *train_state)
            if state_callback:
                wf_state, overflow = state_callback(new_train_state[2]['wf_state'])
            if state_callback and overflow:
                train_state = copy.deepcopy(history.pop(-1))
                train_state[2]['wf_state'] = wf_state
                history.append(copy.deepcopy(train_state))
            elif jnp.isnan(new_train_state[2]['psi'].log).any():
                if rewind and len(history) >= rewind:
                    train_state = history.pop(0)
                else:
                    raise NanError()
            else:
                train_state = new_train_state
                done = True
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
        yield step, train_state, train_stats
