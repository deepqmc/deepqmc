import math
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from .wf.baseline import Baseline


def pretrain(  # noqa: C901
    rng,
    hamil,
    ansatz,
    opt,
    sampler,
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
        steps: an iterable yielding the step numbers for the pretraining.
        sample_size (int): the number of samples to use in a batch.
        baseline_kwargs (dict): optional, additional keyword arguments passed to the
            baseline wave function.
    """
    baseline_init = Baseline.from_mol(sampler.mols, **(baseline_kwargs or {}))

    @hk.without_apply_rng
    @hk.transform
    def baseline(phys_config, return_mos=False):
        return Baseline(hamil.mol, *baseline_init)(phys_config, return_mos)

    init_pc = hamil.init_sample(rng, sampler.mols[0].coords, 1)[0]
    params = ansatz.init(rng, init_pc, False)
    params_baseline = baseline.init(rng, init_pc, False)
    baseline = partial(baseline.apply, params_baseline)

    def loss_fn(params, phys_config):
        n_batch = phys_config.r.shape[0]
        dets = jax.vmap(ansatz.apply, (None, 0, None))(params, phys_config, True)
        dets_target = jax.vmap(baseline, (0, None))(phys_config, True)
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
        return losses.mean(), losses.mean(axis=-1)

    loss_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    if isinstance(opt, optax.GradientTransformation):

        @jax.jit
        def _step(rng, params, opt_state, phys_config):
            (_, losses), grads = loss_and_grad_fn(params, phys_config)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, losses

    else:
        raise NotImplementedError

    smpl_state = sampler.init(rng, baseline, sample_size)
    opt_state = opt.init(params)

    @jax.jit
    def sample_wf(state, rng, select_idxs):
        return sampler.sample(rng, state, baseline, select_idxs)

    for step, rng in zip(steps, hk.PRNGSequence(rng)):
        rng, rng_sample = jax.random.split(rng)
        select_idxs = sampler.select_idxs(sample_size, step)
        smpl_state, phys_config, smpl_stats = sample_wf(
            smpl_state, rng_sample, select_idxs
        )

        params, opt_state, losses = _step(
            rng,
            params,
            opt_state,
            phys_config,
        )
        yield step, params, losses
