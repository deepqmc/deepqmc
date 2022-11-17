import math
from copy import deepcopy
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import kfac_jax
import optax

from .kfacext import GRAPH_PATTERNS
from .wf.base import state_callback
from .wf.baseline import Baseline


def pretrain(  # noqa: C901
    rng,
    hamil,
    ansatz,
    opt,
    sampler,
    state_callback=state_callback,
    *,
    steps,
    sample_size,
    baseline_kwargs=None,
):
    baseline_init = Baseline.from_mol(hamil.mol, **(baseline_kwargs or {}))

    @hk.without_apply_rng
    @hk.transform_with_state
    def ansatz_baseline(rs, return_mos=False):
        return Baseline(hamil.mol, *baseline_init)(rs, return_mos)

    def loss_fn(params, wf_state, r):
        def _loss_fn(params, wf_state, r):
            dets, wf_state = ansatz.apply(params, wf_state, r, True)
            dets_target, _ = baseline({}, r, True)
            repeats = math.ceil(
                math.prod(dets[0].shape[:-2]) / dets_target[0].shape[-3]
            )
            dets_target = (
                jnp.tile(det, (repeats, 1, 1))[: math.prod(dets[0].shape[:-2])]
                for det in (dets_target)
            )
            mos, mos_target = (
                jnp.concatenate((det_up.flatten(), det_down.flatten()), axis=-1)
                for det_up, det_down in (dets, dets_target)
            )
            kfac_jax.register_squared_error_loss(mos)
            loss = ((mos - mos_target) ** 2).mean()
            return loss, wf_state

        loss, wf_state = jax.vmap(_loss_fn, (None, 0, 0))(params, wf_state, r)
        return jnp.mean(loss), (None, wf_state)

    loss_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    if isinstance(opt, optax.GradientTransformation):

        @jax.jit
        def _step(rng, wf_state, params, opt_state, r):
            (loss, (_, wf_state)), grads = loss_and_grad_fn(params, wf_state, r)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return wf_state, params, opt_state, loss

        def init_opt(rng, wf_state, params, r):
            opt_state = opt.init(params)
            return opt_state

    else:

        def _step(rng, wf_state, params, opt_state, r):
            params, opt_state, _, opt_stats = opt.step(
                params,
                opt_state,
                rng,
                func_state=wf_state,
                batch=r,
                momentum=0,
            )
            return opt_stats['aux'], params, opt_state, opt_stats['loss']

        def init_opt(rng, wf_state, params, r):
            opt_state = opt.init(
                params,
                rng,
                r,
                wf_state,
            )
            return opt_state

        opt = opt(
            value_and_grad_func=loss_and_grad_fn,
            l2_reg=0.0,
            value_func_has_aux=True,
            value_func_has_state=True,
            auto_register_kwargs={'graph_patterns': GRAPH_PATTERNS},
            include_norms_in_stats=True,
            estimation_mode='fisher_exact',
            num_burnin_steps=0,
            min_damping=5e-4,
            inverse_update_period=1,
        )

    init_r = hamil.init_sample(rng, sample_size)
    params, wf_state = jax.vmap(ansatz.init, (None, 0, None), (None, 0))(
        rng, init_r, False
    )
    params_baseline, _ = jax.vmap(ansatz_baseline.init, (None, 0, None), (None, 0))(
        rng, init_r, False
    )
    baseline = partial(ansatz_baseline.apply, params_baseline)
    smpl_state = sampler.init(rng, baseline, {}, sample_size, state_callback)
    opt_state = init_opt(rng, wf_state, params, init_r)

    @jax.jit
    def sample_wf(state, rng):
        return sampler.sample(rng, state, baseline)

    for step, rng in zip(steps, hk.PRNGSequence(rng)):
        rng, rng_sample = jax.random.split(rng)
        smpl_state, r, smpl_stats = sample_wf(smpl_state, rng_sample)
        while True:
            wf_state, params_new, opt_state_new, loss = _step(
                rng, wf_state, deepcopy(params), deepcopy(opt_state), r
            )
            wf_state, overflow = state_callback(wf_state)
            if not overflow:
                params, opt_state = params_new, opt_state_new
                break
        yield step, params, loss, {}
