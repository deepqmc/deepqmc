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
        sampler (~deepqmc.sampling.Sampler): the sampler instance to use.
        steps: an iterable yielding the step numbers for the pretraining.
        sample_size (int): the number of samples to use in a batch.
        baseline_kwargs (dict): optional, additional keyword arguments passed to the
            baseline wave function.
    """
    baseline_init = Baseline.from_mol(sampler.mols, **(baseline_kwargs or {}))

    @hk.without_apply_rng
    @hk.transform
    def baseline(phys_conf):
        return Baseline(hamil.mol, None, **baseline_init)(phys_conf)

    init_pc = hamil.init_sample(rng, sampler.mols[0].coords, 1)[0]
    params = ansatz.init(rng, init_pc, False)
    params_baseline = baseline.init(rng, init_pc)
    baseline = partial(baseline.apply, params_baseline)

    def loss_fn(params, phys_config):
        orbs = jax.vmap(ansatz.apply, (None, 0, None))(params, phys_config, True)
        _, n_det, n_up, n_orb_up = orbs[0].shape
        target = jax.vmap(baseline)(phys_config)
        n_det_target = target.shape[-3]
        target = jnp.tile(target, (math.ceil(n_det / n_det_target), 1, 1))[:, :n_det]
        # if the baseline has fewer determinatns than the ansatz targets are repeated
        target = (target[..., :n_up, :n_up], target[..., n_up:, n_up:])
        if n_orb_up != n_up:
            target = (
                jnp.apply_along_axis(jnp.pad, -1, target[0], (0, n_orb_up - n_up)),
                jnp.apply_along_axis(jnp.pad, -1, target[1], (n_up, 0)),
            )
        # in full determinant mode off diagonal elements are pretrained against zero
        losses = jax.tree_util.tree_map(lambda o, t: (o - t) ** 2, orbs, target)
        loss = sum(map(jnp.mean, losses))
        per_sample_losses = sum(map(partial(jnp.mean, axis=(-3, -2, -1)), losses))
        return loss, per_sample_losses

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

    smpl_state = sampler.init(rng, partial(ansatz.apply, params), sample_size)
    opt_state = opt.init(params)

    @jax.jit
    def sample_wf(state, rng, params, select_idxs):
        return sampler.sample(rng, state, partial(ansatz.apply, params), select_idxs)

    for step, rng in zip(steps, hk.PRNGSequence(rng)):
        rng, rng_sample = jax.random.split(rng)
        select_idxs = sampler.select_idxs(sample_size, step)
        smpl_state, phys_config, smpl_stats = sample_wf(
            smpl_state, rng_sample, params, select_idxs
        )

        params, opt_state, losses = _step(
            rng,
            params,
            opt_state,
            phys_config,
        )
        yield step, params, losses
