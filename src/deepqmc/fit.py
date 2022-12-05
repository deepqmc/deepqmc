import logging
from collections import namedtuple
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import kfac_jax
import optax

from .kfacext import make_graph_patterns
from .utils import check_overflow, exp_normalize_mean, masked_mean, tree_norm

__all__ = ()
log = logging.getLogger(__name__)

TrainState = namedtuple('TrainState', 'sampler params opt')


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


def init_fit(rng, hamil, ansatz, sampler, sample_size, state_callback):
    r = hamil.init_sample(rng, sample_size)
    params, wf_state = jax.vmap(ansatz.init, (None, 0), (None, 0))(rng, r)
    smpl_state = sampler.init(
        rng, partial(ansatz.apply, params), wf_state, sample_size, state_callback
    )
    return params, smpl_state


def fit_wf(  # noqa: C901
    rng,
    hamil,
    ansatz,
    opt,
    sampler,
    sample_size,
    steps,
    state_callback=None,
    train_state=None,
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
        #   https://github.com/deepmind/kfac-jax/blob/17831f5a0621b0259c644503556ee7f65acdf0c5/kfac_jax/_src/optimizer.py#L1380-L1383  # noqa: B950
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

    if opt is None:

        @jax.jit
        def _step(_rng_opt, wf_state, params, _opt_state, batch):
            loss, (_, (E_loc, stats)) = loss_fn(params, wf_state, batch)

            return params, None, E_loc, stats

    elif isinstance(opt, optax.GradientTransformation):

        @jax.jit
        def _step(rng, wf_state, params, opt_state, batch):
            (loss, (_, (E_loc, loss_stats))), grads = energy_and_grad_fn(
                params, wf_state, batch
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
                **loss_stats,
            }
            return params, opt_state, E_loc, stats

        def init_opt(rng, wf_state, params, batch):
            opt_state = opt.init(params)
            return opt_state

    else:

        def _step(rng, wf_state, params, opt_state, batch):
            params, opt_state, _, opt_stats = opt.step(
                params,
                opt_state,
                rng,
                func_state=wf_state,
                batch=batch,
                momentum=0,
            )
            stats = {
                'opt/param_norm': opt_stats['param_norm'],
                'opt/grad_norm': opt_stats['precon_grad_norm'],
                'opt/update_norm': opt_stats['update_norm'],
                **opt_stats['aux'][1],
            }
            return params, opt_state, opt_stats['aux'][0], stats

        def init_opt(rng, wf_state, params, batch):
            opt_state = opt.init(
                params,
                rng,
                batch,
                wf_state,
            )
            return opt_state

        opt = opt(
            value_and_grad_func=energy_and_grad_fn,
            l2_reg=0.0,
            value_func_has_aux=True,
            value_func_has_state=True,
            auto_register_kwargs={'graph_patterns': make_graph_patterns()},
            include_norms_in_stats=True,
            estimation_mode='fisher_exact',
            num_burnin_steps=0,
            min_damping=5e-4,
            inverse_update_period=1,
        )

    @partial(check_overflow, state_callback)
    @jax.jit
    def sample_wf(state, rng, params):
        return sampler.sample(rng, state, partial(ansatz.apply, params))

    def train_step(rng, smpl_state, params, opt_state):
        rng_sample, rng_kfac = jax.random.split(rng)
        smpl_state, r, smpl_stats = sample_wf(smpl_state, rng_sample, params)
        weight = exp_normalize_mean(
            smpl_state.get('log_weight', jnp.zeros(sample_size))
        )
        params, opt_state, E_loc, stats = _step(
            rng_kfac, smpl_state['wf'], params, opt_state, (r, weight)
        )
        stats = {
            **smpl_stats,
            **stats,
        }
        return smpl_state, params, opt_state, E_loc, stats

    if train_state:
        smpl_state, params, opt_state = train_state
    else:
        params, smpl_state = init_fit(
            rng, hamil, ansatz, sampler, sample_size, state_callback
        )
        opt_state = None
    if opt is not None and opt_state is None:
        opt_state = init_opt(
            rng, smpl_state['wf'], params, (smpl_state['r'], jnp.ones(sample_size))
        )
    train_state = smpl_state, params, opt_state

    for step, rng in zip(steps, hk.PRNGSequence(rng)):
        *train_state, E_loc, stats = train_step(rng, *train_state)
        yield step, TrainState(*train_state), E_loc, stats
