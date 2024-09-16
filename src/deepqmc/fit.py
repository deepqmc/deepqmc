import operator
from collections.abc import Generator, Iterable
from functools import reduce
from typing import Type

import jax
import jax.numpy as jnp

from .ewm import init_multi_mol_multi_state_ewm
from .loss import LossFunctionFactory
from .observable import ObservableMonitor
from .optimizer import NoOptimizer, Optimizer
from .parallel import (
    local_slice,
    pexp_normalize_mean,
    pmap,
    pmap_pmean,
    replicate_on_devices,
    rng_iterator,
    select_one_device,
    split_on_devices,
)
from .types import Ansatz, DataDict, KeyArray, Stats, TrainState
from .utils import split_dict

__all__ = ()


def fit_wf(  # noqa: C901
    rng: KeyArray,
    hamil,
    ansatz: Ansatz,
    optimizer_factory: Type[Optimizer],
    molecule_idx_sampler,
    sampler,
    steps: Iterable,
    train_state: TrainState,
    loss_function_factory: LossFunctionFactory,
    observable_monitors: list[ObservableMonitor],
) -> Generator[tuple[int, TrainState, jax.Array, Stats, dict]]:
    device_count = jax.device_count()
    loss_fn = loss_function_factory(hamil, ansatz)
    loss_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    opt = optimizer_factory(loss_and_grad_fn)

    @pmap
    def sample_wf(*args):
        return sampler.sample(*args)

    @pmap
    def update_sampler(*args):
        return sampler.update(*args)

    def train_step(
        rng: KeyArray, step: int, data: DataDict, train_state: TrainState
    ) -> tuple[TrainState, jax.Array, Stats]:
        smpl_state, params, opt_state = train_state
        rng_sample, rng_kfac = split_on_devices(rng, 2)
        mol_idxs = molecule_idx_sampler.sample()
        data = jax.tree_util.tree_map(lambda x: x[:, mol_idxs[0]], data)
        smpl_state, phys_conf, smpl_stats = sample_wf(
            rng_sample, smpl_state, params, mol_idxs
        )
        weight = pmap(pexp_normalize_mean)(
            smpl_state['log_weight'][jnp.arange(device_count)[:, None], mol_idxs]
            if 'log_weight' in smpl_state.keys()
            else jnp.zeros(phys_conf.batch_shape)
        )
        params, opt_state, E_loc, ratios, stats = opt.step(
            rng_kfac,
            params,
            opt_state,
            (phys_conf, weight, data),
        )
        # E_loc and ratios have been `all_gather`ed in the loss function
        # because KFAC only returns their mean over the devices
        E_loc = E_loc[0, local_slice()]
        if ratios is not None:
            ratios = ratios[0, local_slice()]
        if not isinstance(opt, NoOptimizer):
            # WF was changed in _step, update psi values stored in smpl_state
            smpl_state = update_sampler(smpl_state, params)
        stats = reduce(
            operator.or_,
            (
                monitor(
                    step, params, phys_conf, smpl_state['elec']['psi'], E_loc, ratios
                )
                for monitor in observable_monitors
            ),
            stats | smpl_stats,
        )
        return TrainState(smpl_state, params, opt_state), mol_idxs, stats

    smpl_state, params, opt_state = train_state
    ewm_state, update_ewm = init_multi_mol_multi_state_ewm(
        shape=(molecule_idx_sampler.n_mols, smpl_state['elec']['r'].shape[-4]),
    )
    std_ewm_state, _ = init_multi_mol_multi_state_ewm(
        shape=(molecule_idx_sampler.n_mols, smpl_state['elec']['r'].shape[-4]),
    )
    data = {
        'energy_ewm': replicate_on_devices(ewm_state.mean),
        'std_ewm': replicate_on_devices(std_ewm_state.mean),
    }

    if opt_state is None:
        rng, rng_sample, rng_opt = split_on_devices(rng, 3)
        idxs = molecule_idx_sampler.sample()
        data = jax.tree_util.tree_map(lambda x: x[:, idxs[0]], data)
        _, init_phys_conf, _ = sample_wf(rng_sample, smpl_state, params, idxs)
        opt_state = opt.init(
            rng_opt,
            params,
            (init_phys_conf, jnp.ones(init_phys_conf.batch_shape), data),
        )
    train_state = TrainState(smpl_state, params, opt_state)

    for step, rng in zip(steps, rng_iterator(rng)):
        train_state, mol_idxs, stats = train_step(rng, step, data, train_state)

        observable_samples, stats = split_dict(stats, lambda k: 'samples' in k)
        stats = select_one_device(pmap_pmean(stats))
        mol_idxs = select_one_device(mol_idxs)

        ewm_state = update_ewm(stats['local_energy/mean'], ewm_state, mol_idxs)
        std_ewm_state = update_ewm(stats['local_energy/std'], std_ewm_state, mol_idxs)

        data = {
            'energy_ewm': replicate_on_devices(ewm_state.mean),
            'std_ewm': replicate_on_devices(std_ewm_state.mean),
        }
        stats |= {
            'energy/ewm': ewm_state.mean[mol_idxs],
            'energy/ewm_error': jnp.sqrt(ewm_state.sqerr[mol_idxs]),
            'energy/std_ewm': std_ewm_state.mean[mol_idxs],
        }

        yield step, train_state, mol_idxs, stats, observable_samples
