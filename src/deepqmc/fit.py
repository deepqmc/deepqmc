from functools import partial

import jax
import jax.numpy as jnp

from deepqmc.loss import create_energy_loss_fn
from deepqmc.optimizer import NoOptimizer

from .parallel import (
    gather_electrons_on_one_device,
    pexp_normalize_mean,
    pmap,
    rng_iterator,
    select_one_device,
    split_on_devices,
)
from .types import TrainState

__all__ = ()


def fit_wf(
    rng,
    hamil,
    ansatz,
    optimizer_factory,
    molecule_idx_sampler,
    sampler,
    electron_batch_size,
    steps,
    train_state,
    *,
    clip_mask_fn=None,
):
    device_count = jax.device_count()
    loss_fn = create_energy_loss_fn(hamil, ansatz, clip_mask_fn)
    opt = optimizer_factory(loss_fn)

    @pmap
    def sample_wf(state, rng, params, idxs):
        return sampler.sample(rng, state, partial(ansatz.apply, params), idxs)

    @pmap
    def update_sampler(state, params):
        return sampler.update(state, partial(ansatz.apply, params))

    def train_step(rng, step, smpl_state, params, opt_state):
        rng_sample, rng_kfac = split_on_devices(rng, 2)
        mol_idxs = molecule_idx_sampler.sample()
        smpl_state, phys_conf, smpl_stats = sample_wf(
            smpl_state, rng_sample, params, mol_idxs
        )
        weight = pmap(pexp_normalize_mean)(
            smpl_state['log_weight'][jnp.arange(device_count)[:, None], mol_idxs]
            if 'log_weight' in smpl_state.keys()
            else jnp.zeros(phys_conf.batch_shape)
        )
        params, opt_state, E_loc, stats = opt.step(
            rng_kfac,
            params,
            opt_state,
            (phys_conf, weight),
        )
        if not isinstance(opt, NoOptimizer):
            # WF was changed in _step, update psi values stored in smpl_state
            smpl_state = update_sampler(smpl_state, params)
        stats = {**stats, **smpl_stats}
        return smpl_state, params, opt_state, E_loc, mol_idxs, stats

    smpl_state, params, opt_state = train_state
    if opt_state is None:
        rng, rng_sample, rng_opt = split_on_devices(rng, 3)
        idxs = molecule_idx_sampler.sample()
        _, init_phys_conf, _ = sample_wf(smpl_state, rng_sample, params, idxs)
        opt_state = opt.init(
            rng_opt,
            params,
            (init_phys_conf, jnp.ones(init_phys_conf.batch_shape)),
        )
    train_state = smpl_state, params, opt_state

    for step, rng in zip(steps, rng_iterator(rng)):
        *train_state, E_loc, mol_idxs, stats = train_step(rng, step, *train_state)
        yield step, TrainState(*train_state), gather_electrons_on_one_device(
            E_loc
        ), select_one_device(mol_idxs), stats
