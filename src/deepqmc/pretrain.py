import math
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from .parallel import (
    gather_electrons_on_one_device,
    rng_iterator,
    select_one_device,
    split_on_devices,
    split_rng_key_to_devices,
)
from .sampling import initialize_sampler_state
from .wf.baseline import Baseline


def pretrain(  # noqa: C901
    rng,
    hamil,
    mols,
    ansatz,
    params,
    opt,
    molecule_idx_sampler,
    sampler,
    *,
    steps,
    electron_batch_size,
    baseline_kwargs=None,
):
    r"""Perform pretraining of the Ansatz to (MC-)SCF orbitals.

    Args:
        rng (~deepqmc.types.RNGSeed): key used for PRNG.
        hamil (~deepqmc.hamil.qc.MolecularHamiltonian): hamiltonian of the molecule.
        ansatz (~deepqmc.wf.WaveFunction): the wave function Ansatz.
        params (dict): the (initial) parameters of the Ansatz.
        opt (``optax`` optimizers): the optimizer.
        molecule_idx_sampler (~deepqmc.samplint.MoleculeIdxSampler): an object that
            iterates (samples) the indices of the molecule dataset.
        sampler (~deepqmc.sampling.Sampler): the sampler instance to use.
        steps: an iterable yielding the step numbers for the pretraining.
        electron_batch_size (int): the number of electron samples to use in a batch.
        baseline_kwargs (dict): optional, additional keyword arguments passed to the
            baseline wave function.
    """
    partial_baseline = Baseline.from_mol(mols, hamil, **(baseline_kwargs or {}))

    @hk.without_apply_rng
    @hk.transform
    def baseline(phys_conf):
        return partial_baseline(hamil, None)(phys_conf)

    rng, rng_hamil, rng_baseline = jax.random.split(rng, 3)
    init_pc = hamil.init_sample(rng_hamil, mols[0].coords, 1)[0]
    params_baseline = baseline.init(rng_baseline, init_pc)
    baseline = partial(baseline.apply, params_baseline)

    def loss_fn(params, phys_config):
        orbs = jax.vmap(jax.vmap(ansatz.apply, (None, 0, None)), (None, 0, None))(
            params, phys_config, True
        )
        *_, n_det, n_up, n_orb_up = orbs[0].shape
        target = jax.vmap(jax.vmap(baseline))(phys_config)
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

        def _step(rng, params, opt_state, phys_config):
            (_, per_sample_losses), grads = loss_and_grad_fn(params, phys_config)
            grads = jax.lax.pmean(grads, 'device_axis')
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, per_sample_losses

    else:
        raise NotImplementedError

    rng, rng_smpl_init = split_on_devices(split_rng_key_to_devices(rng), 2)
    smpl_state = initialize_sampler_state(
        rng_smpl_init, sampler, ansatz, params, electron_batch_size
    )
    opt_state = jax.pmap(opt.init)(params)

    def sample_wf(state, rng, params, idxs):
        return sampler.sample(rng, state, partial(ansatz.apply, params), idxs)

    @partial(jax.pmap, axis_name='device_axis')
    def pretrain_step(rng, params, smpl_state, opt_state, mol_idxs):
        rng, rng_sample = jax.random.split(rng)
        smpl_state, phys_config, smpl_stats = sample_wf(
            smpl_state, rng_sample, params, mol_idxs
        )

        params, opt_state, per_sample_losses = _step(
            rng,
            params,
            opt_state,
            phys_config,
        )
        return params, opt_state, per_sample_losses

    for step, rng in zip(steps, rng_iterator(rng)):
        mol_idxs = molecule_idx_sampler.sample()
        params, opt_state, per_sample_losses = pretrain_step(
            rng, params, smpl_state, opt_state, mol_idxs
        )
        yield step, params, gather_electrons_on_one_device(
            per_sample_losses
        ), select_one_device(mol_idxs)
