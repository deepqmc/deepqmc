import math
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from .parallel import (
    gather_on_one_device,
    replicate_on_devices,
    rng_iterator,
    select_one_device,
    split_on_devices,
    split_rng_key_to_devices,
)
from .wf.baseline import Baseline


def pretrain(  # noqa: C901
    rng,
    hamil,
    mols,
    ansatz,
    params,
    opt,
    molecule_sampler,
    sampler,
    *,
    steps,
    sample_size,
    baseline_kwargs=None,
):
    r"""Perform pretraining of the Ansatz to (MC-)SCF orbitals.

    Args:
        rng (~deepqmc.types.RNGSeed): key used for PRNG.
        hamil (~deepqmc.hamil.qc.MolecularHamiltonian): hamiltonian of the molecule.
        ansatz (~deepqmc.wf.WaveFunction): the wave function Ansatz.
        params (dict): the (initial) parameters of the Ansatz.
        opt (``optax`` optimizers): the optimizer.
        sampler (~deepqmc.sampling.Sampler): the sampler instance to use.
        steps: an iterable yielding the step numbers for the pretraining.
        sample_size (int): the number of samples to use in a batch.
        baseline_kwargs (dict): optional, additional keyword arguments passed to the
            baseline wave function.
    """
    partial_baseline = Baseline.from_mol(mols, **(baseline_kwargs or {}))

    @hk.without_apply_rng
    @hk.transform
    def baseline(phys_conf, mol_idx):
        return partial_baseline(hamil.mol, None)(phys_conf, mol_idx)

    rng, rng_hamil, rng_baseline = jax.random.split(rng, 3)
    init_pc = hamil.init_sample(rng_hamil, mols[0].coords, 1)[0]
    params_baseline = baseline.init(rng_baseline, init_pc, jnp.array(0))
    baseline = partial(baseline.apply, params_baseline)

    def loss_fn(params, phys_config, idxs):
        orbs = jax.vmap(jax.vmap(ansatz.apply, (None, 0, None)), (None, 0, None))(params, phys_config, True)
        *_, n_det, n_up, n_orb_up = orbs[0].shape
        target = jax.vmap(jax.vmap(baseline, (0, None)))(phys_config, idxs)
        n_det_target = target.shape[-3]
        target = jnp.tile(target, (math.ceil(n_det / n_det_target), 1, 1))[:, :, :n_det]
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

        def _step(rng, params, opt_state, phys_config, idxs):
            (_, losses), grads = loss_and_grad_fn(params, phys_config, idxs)
            grads = jax.lax.pmean(grads, 'device_axis')
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, losses

    else:
        raise NotImplementedError

    rng, rng_smpl_init = split_on_devices(split_rng_key_to_devices(rng))
    wf = partial(ansatz.apply, select_one_device(params))
    sample_initializer = partial(
        sampler.init, wf=wf, electron_batch_size=sample_size // jax.device_count()
    )
    smpl_state = jax.pmap(sample_initializer)(rng_smpl_init)

    opt_state = jax.pmap(opt.init)(params)

    def sample_wf(state, rng, params, idxs):
        return sampler.sample(rng, state, partial(ansatz.apply, params), idxs)

    @partial(jax.pmap, axis_name='device_axis')
    def pretrain_step(rng, params, smpl_state, opt_state, idxs):
        rng, rng_sample = jax.random.split(rng)
        smpl_state, phys_config, smpl_stats = sample_wf(
            smpl_state, rng_sample, params, idxs
        )

        params, opt_state, losses = _step(
            rng,
            params,
            opt_state,
            phys_config,
            idxs,
        )
        return params, opt_state, losses

    for step, rng in zip(steps, rng_iterator(rng)):
        idxs = molecule_sampler.sample()
        params, opt_state, losses = pretrain_step(
            rng, params, smpl_state, opt_state, idxs
        )
        yield step, params, gather_on_one_device(
            losses, flatten_device_axis=False
        )
