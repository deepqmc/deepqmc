import math
from copy import deepcopy
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import optax

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
    r"""Perform pretraining of the Ansatz to (MC-)SCF orbitals.

    Args:
        rng (~deepqmc.types.RNGSeed): key used for PRNG.
        hamil (~deepqmc.hamil.Hamiltonian): the Hamiltonian of the physical system.
        ansatz (~deepqmc.wf.WaveFunction): the wave function Ansatz.
        opt (``optax`` optimizer): the optimizer to use.
        sampler (~deepqmc.sampling.Sampler): the sampler instance to use.
        state_callback (Callable): optional, a function processing the :class:`haiku`
            state of the wave function Ansatz.
        steps: an iterable yielding the step numbers for the pretraining.
        sample_size (int): the number of samples to use in a batch.
        baseline_kwargs (dict): optional, additional keyword arguments passed to the
            baseline wave function.
    """
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
            loss = ((mos - mos_target) ** 2).mean()
            return loss, wf_state

        loss, wf_state = jax.vmap(_loss_fn, (None, 0, 0))(params, wf_state, r)
        return jnp.mean(loss), wf_state

    loss_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    if isinstance(opt, optax.GradientTransformation):

        @jax.jit
        def _step(rng, wf_state, params, opt_state, r):
            (loss, wf_state), grads = loss_and_grad_fn(params, wf_state, r)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return wf_state, params, opt_state, loss

    else:
        raise NotImplementedError

    init_r = hamil.init_sample(rng, sample_size)
    params, wf_state = jax.vmap(ansatz.init, (None, 0, None), (None, 0))(
        rng, init_r, False
    )
    params_baseline, _ = jax.vmap(ansatz_baseline.init, (None, 0, None), (None, 0))(
        rng, init_r, False
    )
    baseline = partial(ansatz_baseline.apply, params_baseline)
    smpl_state = sampler.init(rng, baseline, {}, sample_size, state_callback)
    opt_state = opt.init(params)

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
        yield step, params, loss
