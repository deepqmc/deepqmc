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
    baseline_inits = [
        Baseline.from_mol(hamil.mol, R, **(baseline_kwargs or {}))
        for R in sampler.nuclear_configurations()
    ]

    def ansatz_baseline(baseline_init, phys_config, return_mos=False):
        return Baseline(hamil.mol, *baseline_init)(phys_config, return_mos)

    baselines = [
        hk.without_apply_rng(
            hk.transform_with_state(partial(ansatz_baseline, baseline_init))
        )
        for baseline_init in baseline_inits
    ]

    init_pc = hamil.init_sample(
        rng, next(sampler.nuclear_configurations()), sample_size
    )
    params, wf_state = jax.vmap(ansatz.init, (None, 0, None), (None, 0))(
        rng, init_pc, False
    )
    params_baselines = [baseline.init(rng, init_pc, False)[0] for baseline in baselines]
    baselines = [
        partial(baseline.apply, params_baseline)
        for baseline, params_baseline in zip(baselines, params_baselines)
    ]

    def loss_fn(params, wf_state, phys_config, select_idxs):
        n_batch = phys_config.r.shape[0]
        dets, wf_state = jax.vmap(ansatz.apply, (None, 0, 0, None))(
            params, wf_state, phys_config, True
        )
        dets_targets = [
            jax.vmap(baseline, (0, 0, None))({}, phys_config, True)[0]
            for baseline in baselines
        ]
        dets_target = sampler.join_configs(dets_targets, select_idxs)
        repeats = math.ceil(dets[0].shape[-3] / dets_target[0].shape[-3])
        dets_target = (
            jnp.tile(det, (1, repeats, 1, 1))[:, : dets[0].shape[-3]]
            for det in dets_target
        )
        mos, mos_target = (
            jnp.concatenate(
                (det_up.reshape(n_batch, -1), det_down.reshape(n_batch, -1)), axis=-1
            )
            for det_up, det_down in (dets, dets_target)
        )
        losses = (mos - mos_target) ** 2
        return losses.mean(), (wf_state, losses.mean(axis=-1))

    loss_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    if isinstance(opt, optax.GradientTransformation):

        @jax.jit
        def _step(rng, wf_state, params, opt_state, phys_config, select_idxs):
            (_, (wf_state, losses)), grads = loss_and_grad_fn(
                params, wf_state, phys_config, select_idxs
            )
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return wf_state, params, opt_state, losses

    else:
        raise NotImplementedError

    smpl_state = sampler.init(rng, baselines, sample_size)
    opt_state = opt.init(params)

    @jax.jit
    def sample_wf(state, rng, select_idxs):
        return sampler.sample(rng, state, baselines, select_idxs)

    for step, rng in zip(steps, hk.PRNGSequence(rng)):
        rng, rng_sample = jax.random.split(rng)
        select_idxs = sampler.select_idxs(sample_size, smpl_state)
        smpl_state, phys_config, smpl_stats = sample_wf(
            smpl_state, rng_sample, select_idxs
        )

        while True:
            wf_state, params_new, opt_state_new, losses = _step(
                rng,
                wf_state,
                deepcopy(params),
                deepcopy(opt_state),
                phys_config,
                select_idxs,
            )
            wf_state, overflow = state_callback(wf_state)
            if not overflow:
                params, opt_state = params_new, opt_state_new
                break
        yield step, params, losses
