import logging
import os
import time
from collections.abc import Callable, Sequence
from functools import partial
from itertools import count
from typing import Optional, Type

import jax
import jax.numpy as jnp
import optax
from tqdm.auto import tqdm, trange
from uncertainties import ufloat

from .conf.custom_resolvers import process_idx_suffix
from .ewm import init_multi_mol_multi_state_ewm
from .exceptions import NanError, TrainingBlowup, TrainingCrash
from .fit import fit_wf
from .hamil import MolecularHamiltonian
from .log import CheckpointStore, H5Logger, MetricLogger, TensorboardMetricLogger
from .loss.clip import median_log_squeeze_and_mask
from .loss.loss_function import LossFunctionFactory, create_loss_fn
from .molecule import Molecule
from .observable import ObservableMonitor, default_observable_monitors
from .optimizer import NoOptimizer
from .parallel import pmap_pmean, split_on_devices, split_rng_key_to_devices
from .physics import pairwise_self_distance
from .pretrain.pretraining import pretrain
from .pretrain.pyscfext import compute_scf_solution
from .sampling import (
    MoleculeIdxSampler,
    MultiNuclearGeometrySampler,
    equilibrate,
    initialize_sampler_state,
)
from .types import Ansatz, KeyArray, TrainState
from .wf.base import init_wf_params

__all__ = ['train']

log = logging.getLogger(__name__)


def train(  # noqa: C901
    hamil,
    ansatz: Ansatz,
    opt,
    sampler_factory: Callable[
        [KeyArray, MolecularHamiltonian, Ansatz, list[Molecule], int, int],
        tuple[MoleculeIdxSampler, MultiNuclearGeometrySampler],
    ],
    steps: int,
    seed: int,
    electron_batch_size: int,
    molecule_batch_size: int = 1,
    electronic_states: int = 1,
    mols: Optional[list[Molecule]] = None,
    workdir: Optional[str] = None,
    train_state: Optional[TrainState] = None,
    init_step: int = 0,
    max_restarts: int = 3,
    max_eq_steps: int = 1000,
    eq_allow_early_stopping: bool = True,
    pretrain_steps: Optional[int] = None,
    pretrain_kwargs: Optional[dict] = None,
    chkpt_constructor: Optional[Type[CheckpointStore]] = None,
    metric_logger_constructor: Optional[Type[MetricLogger]] = None,
    h5_logger_constructor: Optional[Type[H5Logger]] = None,
    merge_keys: Optional[list[str]] = None,
    loss_function_factory: Optional[LossFunctionFactory] = None,
    observable_monitors: Optional[list[ObservableMonitor]] = None,
):
    r"""Train or evaluate a JAX wave function model.

    It initializes and equilibrates the MCMC sampling of the wave function ansatz,
    then optimizes or samples it using the variational principle. It optionally
    saves checkpoints and rewinds the training/evaluation if an error is encountered.
    If an optimizer is supplied, the Ansatz is optimized, otherwise the Ansatz is
    only sampled.

    Args:
        hamil (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the
            physical system.
        ansatz (~deepqmc.types.Ansatz): the wave function Ansatz.
        opt (``kfac_jax`` or ``optax`` optimizers | :data:`None`):
            the optimizer. Possible values are:

            - :class:`kfac_jax.Optimizer`:
                the partially initialized KFAC optimizer is used
            - an :data:`optax` optimizer instance:
                the supplied :data:`optax` optimizer is used.
            - :data:`None`:
                no optimizer is used, e.g. the evaluation of the Ansatz is performed.
        sampler_factory (~collections.abc.Callable): a function that returns a Sampler
            instance
        steps (int): number of optimization steps.
        seed (int): the seed used for PRNG.
        electron_batch_size (int): the number of electron samples considered in a batch
        molecule_batch_size (int): optional, the number of molecules considered in a
            batch. Only needed for transferable training.
        electronic_states (int): optional, the number of electronic states to consider.
        mols (list[~deepqmc.molecule.Molecule]): optional, a sequence of molecules
            to consider for transferable training. If None the default molecule from
            hamil is used.
        workdir (str): optional, path, where results should be saved.
        train_state (~deepqmc.types.TrainState): optional, training checkpoint to
            restore training or run evaluation.
        init_step (int): optional, initial step index, useful if
            calculation is restarted from checkpoint saved on disk.
        max_restarts (int): optional, the maximum number of times the training is
            retried before a :class:`NaNError` is raised.
        max_eq_steps (int): optional, maximum number of equilibration steps if not
            detected earlier.
        eq_allow_early_stopping (bool): default ``True``, whether to allow the
            equilibration to stop early when the equilibration criterion has been met.
        pretrain_steps (int): optional, the number of pretraining steps wrt. to the
            Baseline wave function obtained with pyscf.
        pretrain_kwargs (dict): optional, extra arguments for pretraining.
        chkpt_constructor (~typing.Type[~deepqmc.log.CheckpointStore]): optional, an
            object that saves training checkpoints to the working directory.
        metric_logger_constructor (~typing.Type[~deepqmc.log.MetricLogger]): optional,
            an object that consumes metric logging information. If not specified, the
            default :class:`~deepqmc.log.TensorboardMetricLogger` is used to create
            tensorboard logs.
        h5_logger_constructor (~typing.Type[~deepqmc.log.H5Logger]): optional, an object
            that consumes metric logging information. If not specified, the default
            :class:`~deepqmc.log.H5Logger` is used to write comprehensive training
            statistics to an h5file.
        merge_keys (list[str]): optional, list of strings for selecting parameters to be
            shared across electronic states. Matching merge keys with (substrings of)
            parameter keys.
        loss_function_factory (~deepqmc.loss.loss_function.LossFunctionFactory):
            optional, a callable returning a suitable loss function for the
            optimization.
        observable_monitors (list[~deepqmc.observable.ObservableMonitor]): optional,
            list of observable monitors to be evaluated during training or evaluation.
    """
    mode = 'evaluation' if opt is None else 'training'
    rng = jax.random.PRNGKey(seed + jax.process_index())
    rng, rng_smpl = jax.random.split(rng)
    mols = mols if isinstance(mols, Sequence) else [hamil.mol]
    molecule_idx_sampler, sampler = sampler_factory(
        rng_smpl,
        hamil,
        ansatz,
        mols,
        electronic_states,
        molecule_batch_size,
    )
    opt = opt or NoOptimizer
    chkpts = None
    if workdir:
        workdir = os.path.join(workdir, mode + process_idx_suffix())
        os.makedirs(workdir, exist_ok=True)
        chkpts = (chkpt_constructor or CheckpointStore)(workdir)
        log.debug('Setting up metric_logger...')
        metric_logger = (metric_logger_constructor or TensorboardMetricLogger)(
            workdir, molecule_batch_size
        )
        log.debug('Setting up h5_logger...')
        h5_logger = (h5_logger_constructor or H5Logger)(
            workdir,
            init_step=init_step,
            aux_data={f'mol-{i}': m.coords for i, m in enumerate(mols)},
        )
        init_time = time.time()
    else:
        metric_logger = None
        h5_logger = None
        init_time = None

    pbar = None
    try:
        if train_state:
            log.info(
                {
                    'training': f'Restart training from step {init_step}',
                    'evaluation': 'Start evaluation',
                }[mode]
            )
            params = train_state.params
        else:
            rng, rng_init = jax.random.split(rng)
            params = init_wf_params(
                rng_init, hamil, ansatz, electronic_states, merge_keys=merge_keys
            )
            if pretrain_steps and mode == 'training':
                log.info('Pretraining wrt. baseline wave function')
                rng, rng_pretrain = jax.random.split(rng)
                pretrain_kwargs = pretrain_kwargs or {}
                pretrain_dataset = compute_scf_solution(
                    mols,
                    hamil,
                    electronic_states,
                    workdir=pretrain_kwargs.pop('pyscf_chkpt_path', None) or workdir,
                    **pretrain_kwargs.pop('scf_kwargs', {}),
                )
                opt_pretrain = getattr(optax, pretrain_kwargs.pop('opt', 'adam'))(
                    **pretrain_kwargs.pop('opt_kwargs', {'learning_rate': 3.0e-4})
                )
                ewm_state, update_ewm = init_multi_mol_multi_state_ewm(
                    shape=(len(mols), electronic_states), decay_alpha=1.0
                )
                mse_rep = None
                _, rng_pretrain_smpl_init = split_on_devices(
                    split_rng_key_to_devices(rng), 2
                )
                pretrain_smpl_state = initialize_sampler_state(
                    rng_pretrain_smpl_init, sampler, params, electron_batch_size, mols
                )
                pbar = tqdm(range(pretrain_steps), desc='pretrain', disable=None)
                for step, params, per_sample_losses, mol_idxs in pretrain(  # noqa: B007
                    rng_pretrain,
                    hamil,
                    ansatz,
                    params,
                    opt_pretrain,
                    molecule_idx_sampler,
                    sampler,
                    pretrain_smpl_state,
                    pretrain_dataset,
                    steps=pbar,
                ):  # noqa: B007
                    per_mol_state_losses = per_sample_losses.mean(axis=-1)
                    ewm_state = update_ewm(per_mol_state_losses, ewm_state, mol_idxs)
                    pretrain_stats = {
                        'MSE': per_mol_state_losses,
                        'MSE/ewm': ewm_state.mean,
                    }
                    mse_rep = '|'.join(
                        '(' + '|'.join(f'{mses:0.2e}' for mses in msem) + ')'
                        for msem in ewm_state.mean
                    )
                    pbar.set_postfix(MSE=mse_rep)
                    if metric_logger:
                        metric_logger.update(
                            step, pretrain_stats, {}, mol_idxs, prefix='pretraining'
                        )
                log.info(f'Pretraining completed with MSE = {mse_rep}')

        rng = split_rng_key_to_devices(rng)
        if train_state is None or train_state.sampler is None:
            rng, rng_eq, rng_smpl_init = split_on_devices(rng, 3)
            smpl_state = initialize_sampler_state(
                rng_smpl_init, sampler, params, electron_batch_size, mols
            )
            log.info('Equilibrating sampler...')
            pbar = tqdm(
                count() if max_eq_steps is None else range(max_eq_steps),
                desc='equilibrate sampler',
                disable=None,
            )
            for step, smpl_state, mol_idxs, smpl_stats in equilibrate(  # noqa: B007
                rng_eq,
                params,
                molecule_idx_sampler,
                sampler,
                smpl_state,
                lambda phys_conf: pmap_pmean(
                    pairwise_self_distance(phys_conf.r)
                ).mean(),
                pbar,
                block_size=10,
                allow_early_stopping=eq_allow_early_stopping,
            ):
                tau_rep = '|'.join(
                    '(' + '|'.join(f'{taus:.3f}' for taus in taum) + ')'
                    for taum in smpl_state['elec']['tau'].mean(axis=0)
                )
                pbar.set_postfix(tau=tau_rep)
                if metric_logger:
                    metric_logger.update(
                        step, {}, smpl_stats, mol_idxs, prefix='equilibration'
                    )
            pbar.close()
            train_state = TrainState(smpl_state, params, None)
            if workdir and mode == 'training':
                assert chkpts
                chkpts.update(init_step, train_state)
            log.info(f'Start {mode}')
        observable_monitors = observable_monitors or default_observable_monitors()
        loss_function_factory = loss_function_factory or partial(
            create_loss_fn, clip_mask_fn=median_log_squeeze_and_mask
        )
        best_ene = None
        step = init_step
        ewm_energies = len(mols) * [electronic_states * [ufloat(jnp.nan, 1)]]
        for attempt in range(max_restarts):
            try:
                pbar = trange(
                    init_step,
                    steps,
                    initial=init_step,
                    total=steps,
                    desc=mode,
                    disable=None,
                )
                for (
                    step,
                    train_state,
                    mol_idxs,
                    stats,
                    observable_samples,
                ) in fit_wf(  # noqa: B007
                    rng,
                    hamil,
                    ansatz,
                    opt,
                    molecule_idx_sampler,
                    sampler,
                    pbar,
                    train_state,
                    loss_function_factory,
                    observable_monitors=[
                        monitor.finalize(hamil, ansatz.apply)
                        for monitor in observable_monitors
                    ],
                ):
                    ewm_energies, best_ene = update_progress(
                        pbar, best_ene, ewm_energies, mol_idxs, stats
                    )
                    if jnp.isnan(observable_samples['psi/samples']['log']).any():
                        raise NanError()
                    if workdir:
                        assert init_time is not None
                        assert h5_logger is not None
                        if mode == 'training':
                            assert chkpts
                            # the convention is that chkpt-i contains the step i-1 -> i
                            chkpts.update(
                                step + 1,
                                train_state,
                                stats['local_energy/std'].mean(),
                            )
                        if metric_logger:
                            metric_logger.update(step, stats, {}, mol_idxs)
                        observable_samples |= {
                            'mol_idxs': mol_idxs,
                            'step': step,
                            'time': time.time() - init_time,
                            **stats,
                        }
                        h5_logger.update(observable_samples)
                log.info(f'The {mode} has been completed!')
                return train_state
            except (NanError, TrainingBlowup) as e:
                if pbar:
                    pbar.close()
                log.warn(f'Restarting due to {type(e).__name__}...')
                if attempt < max_restarts and chkpts is not None:
                    init_step, train_state = chkpts.last
                    (rng,) = split_on_devices(rng, 1)
        log.warn(
            f'The {mode} has crashed before all steps were completed ({step}/{steps})!'
        )
        raise TrainingCrash(train_state)
    finally:
        if pbar:
            pbar.close()
        if chkpts:
            chkpts.close()
        if metric_logger:
            metric_logger.close()
        if h5_logger:
            h5_logger.close()


def update_progress(pbar, best_ene, ewm_energies, mol_idxs, stats):
    r"""Update the tqdm progress bar, maybe print progress message."""
    for i, mol_idx in enumerate(mol_idxs):
        ewm_energies[mol_idx] = [
            ufloat(mean, sqerr)
            for (mean, sqerr) in zip(
                stats['energy/ewm'][i], stats['energy/ewm_error'][i]
            )
        ]
    energies = '|'.join(
        '(' + '|'.join(f'{es:S}' for es in em) + ')' for em in ewm_energies
    )
    pbar.set_postfix(E=energies)
    if best_ene is None or jnp.any(
        jnp.array(jax.tree_map(lambda x, y: x.s < 0.5 * y.s, ewm_energies, best_ene))
    ):
        best_ene = ewm_energies
        log.info(f'Progress: {pbar.n + 1}/{pbar.total}, energy = {energies}')
    return ewm_energies, best_ene
