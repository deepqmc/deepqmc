import logging
import os
from functools import partial
from itertools import count

import h5py
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm, trange
from uncertainties import ufloat

from deepqmc.optimizer import construct_optimizer

from .ewm import init_ewm
from .fit import fit_wf
from .log import CheckpointStore, H5LogTable, TensorboardMetricLogger
from .parallel import (
    gather_electrons_on_one_device,
    select_one_device,
    split_on_devices,
    split_rng_key_to_devices,
)
from .physics import pairwise_self_distance
from .pretrain import pretrain
from .sampling import equilibrate, initialize_sampler_state, initialize_sampling
from .wf.base import init_wf_params

__all__ = ['train']

log = logging.getLogger(__name__)


class NanError(Exception):
    def __init__(self):
        super().__init__()


class TrainingCrash(Exception):
    def __init__(self, train_state):
        super().__init__()
        self.train_state = train_state


def train(  # noqa: C901
    hamil,
    ansatz,
    opt,
    sampler,
    steps,
    seed,
    electron_batch_size,
    molecule_batch_size=1,
    mols=None,
    workdir=None,
    train_state=None,
    init_step=0,
    max_restarts=3,
    max_eq_steps=1000,
    pretrain_steps=None,
    pretrain_kwargs=None,
    pretrain_sampler=None,
    opt_kwargs=None,
    fit_kwargs=None,
    chkptdir=None,
    chkpts_kwargs=None,
    metric_logger=None,
):
    r"""Train or evaluate a JAX wave function model.

    It initializes and equilibrates the MCMC sampling of the wave function ansatz,
    then optimizes or samples it using the variational principle. It optionally
    saves checkpoints and rewinds the training/evaluation if an error is encountered.
    If an optimizer is supplied, the Ansatz is optimized, otherwise the Ansatz is
    only sampled.

    Args:
        hamil (~deepqmc.hamil.Hamiltonian): the Hamiltonian of the physical system.
        ansatz (~deepqmc.wf.WaveFunction): the wave function Ansatz.
        opt (``kfac_jax`` or ``optax`` optimizers, :class:`str` or :data:`None`):
            the optimizer. Possible values are:

            - :class:`kfac_jax.Optimizer`: the partially initialized KFAC optimizer
                is used
            - an :data:`optax` optimizer instance: the supplied :data:`optax`
                optimizer is used.
            - :class:`str`: the name of the optimizer to use (:data:`'kfac'` or an
                :data:`optax` optimizer name). Arguments to the optimizer can be
                passed in :data:`opt_kwargs`.
            - :data:`None`: no optimizer is used, e.g. the evaluation of the Ansatz
                is performed.
        sampler (~deepqmc.sampling.Sampler): a sampler instance
        steps (int): number of optimization steps.
        seed (int): the seed used for PRNG.
        electron_batch_size (int): the number of electron samples considered in a batch
        molecule_batch_size (int): optional, the number of molecules considered in a
            batch. Only needed for transferable training.
        mols (Sequence(~deepqmc.molecule.Molecule)): optional, a sequence of molecules
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
        pretrain_steps (int): optional, the number of pretraining steps wrt. to the
            Baseline wave function obtained with pyscf.
        pretrain_kwargs (dict): optional, extra arguments for pretraining.
        opt_kwargs (dict): optional, extra arguments passed to the optimizer.
        fit_kwargs (dict): optional, extra arguments passed to the :func:`~.fit.fit_wf`
            function.
        chkptdir (str): optional, path, where checkpoints should be saved. Checkpoints
            are only saved if :data:`workdir` is not :data:`None`. Default:
            data:`workdir`.
        chkpts_kwargs (dict): optional, extra arguments for checkpointing.
        metric_logger: optional, an object that consumes metric logging information.
            If not specified, the default `~.log.TensorboardMetricLogger` is used
            to create tensorboard logs.
    """
    mode = 'evaluation' if opt is None else 'training'
    rng = jax.random.PRNGKey(seed)
    rng, rng_smpl = jax.random.split(rng)
    mols, molecule_idx_sampler, sampler, pretrain_sampler = initialize_sampling(
        rng_smpl,
        hamil,
        mols,
        sampler,
        pretrain_sampler,
        electron_batch_size,
        molecule_batch_size,
    )
    opt = construct_optimizer(opt, opt_kwargs)
    if workdir:
        workdir = os.path.join(workdir, mode)
        chkptdir = os.path.join(chkptdir, mode) if chkptdir else workdir
        os.makedirs(workdir, exist_ok=True)
        os.makedirs(chkptdir, exist_ok=True)
        chkpts = CheckpointStore(chkptdir, **(chkpts_kwargs or {}))
        metric_logger = (metric_logger or TensorboardMetricLogger)(
            workdir, len(sampler)
        )
        log.debug('Setting up HDF5 file...')
        h5file = h5py.File(os.path.join(workdir, 'result.h5'), 'a', libver='v110')
        h5file.swmr_mode = True
        table = H5LogTable(h5file)
        table.resize(init_step)
        h5file.flush()

    pbar = None
    try:
        if train_state:
            log.info(
                {
                    'training': f'Restart training from step {init_step}',
                    'evaluation': 'Start evaluation',
                }[mode]
            )
            params = train_state[1]
        else:
            rng, rng_init = jax.random.split(rng)
            params = init_wf_params(rng_init, hamil, ansatz)
            if pretrain_steps and mode == 'training':
                log.info('Pretraining wrt. baseline wave function')
                rng, rng_pretrain = jax.random.split(rng)
                pretrain_kwargs = pretrain_kwargs or {}
                opt_pretrain = construct_optimizer(
                    pretrain_kwargs.pop('opt', 'adamw'),
                    pretrain_kwargs.pop('opt_kwargs', None),
                    wrap=False,
                )
                ewm_state, update_ewm = init_ewm(decay_alpha=1.0)
                ewm_states = len(pretrain_sampler) * [ewm_state]
                pbar = tqdm(range(pretrain_steps), desc='pretrain', disable=None)
                for step, params, per_sample_losses, mol_idxs in pretrain(  # noqa: B007
                    rng_pretrain,
                    hamil,
                    mols,
                    ansatz,
                    params,
                    opt_pretrain,
                    molecule_idx_sampler,
                    pretrain_sampler,
                    steps=pbar,
                    electron_batch_size=electron_batch_size,
                    baseline_kwargs=pretrain_kwargs.pop('baseline_kwargs', {}),
                ):
                    per_mol_losses = per_sample_losses.mean(axis=1)
                    ewm_means = []
                    for loss, mol_idx in zip(per_mol_losses, mol_idxs):
                        ewm_states[mol_idx] = update_ewm(loss, ewm_states[mol_idx])
                        ewm_means.append(ewm_states[mol_idx].mean)
                    pretrain_stats = {
                        'MSE': per_mol_losses,
                        'MSE/ewm': jnp.array(ewm_means),
                    }
                    mse_rep = '|'.join(
                        f'{ewm.mean if ewm.mean is not None else jnp.nan:0.2e}'
                        for ewm in ewm_states
                    )
                    pbar.set_postfix(MSE=mse_rep)
                    if metric_logger:
                        metric_logger.update(
                            step, pretrain_stats, {}, mol_idxs, prefix='pretraining'
                        )
                log.info(f'Pretraining completed with MSE = {mse_rep}')

        rng = split_rng_key_to_devices(rng)
        if not train_state or train_state[0] is None:
            rng, rng_eq, rng_smpl_init = split_on_devices(rng, 3)
            smpl_state = initialize_sampler_state(
                rng_smpl_init, sampler, ansatz, params, electron_batch_size
            )
            log.info('Equilibrating sampler...')
            pbar = tqdm(
                count() if max_eq_steps is None else range(max_eq_steps),
                desc='equilibrate sampler',
                disable=None,
            )
            for step, smpl_state, mol_idxs, smpl_stats in equilibrate(  # noqa: B007
                rng_eq,
                partial(ansatz.apply, select_one_device(params)),
                molecule_idx_sampler,
                sampler,
                smpl_state,
                lambda phys_conf: pairwise_self_distance(phys_conf.r).mean(),
                pbar,
                block_size=10,
            ):
                tau_rep = '|'.join(
                    f'{tau:.3f}' for tau in smpl_state['tau'].mean(axis=0)
                )
                pbar.set_postfix(tau=tau_rep)
                if metric_logger:
                    metric_logger.update(
                        step, {}, smpl_stats, mol_idxs, prefix='equilibration'
                    )
            pbar.close()
            train_state = smpl_state, params, None
            if workdir and mode == 'training':
                chkpts.update(init_step, train_state)
            log.info(f'Start {mode}')
        best_ene = None
        ewm_state, update_ewm = init_ewm()
        ewm_states = len(sampler) * [ewm_state]
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
                for step, train_state, E_loc, mol_idxs, stats in fit_wf(  # noqa: B007
                    rng,
                    hamil,
                    ansatz,
                    opt,
                    molecule_idx_sampler,
                    sampler,
                    electron_batch_size,
                    pbar,
                    train_state,
                    **(fit_kwargs or {}),
                ):
                    per_mol_energy = E_loc.mean(axis=1)
                    ewm_energies = []
                    for energy, mol_idx in zip(per_mol_energy, mol_idxs):
                        ewm_states[mol_idx] = update_ewm(energy, ewm_states[mol_idx])
                        ewm_energies.append(ewm_states[mol_idx].mean)
                    ene = [
                        (
                            ufloat(ewm.mean, jnp.sqrt(ewm.sqerr))
                            if ewm.mean
                            else ufloat(jnp.nan, 0)
                        )
                        for ewm in ewm_states
                    ]
                    if all(e.s for e in ene):
                        energies = '|'.join(f'{e:S}' for e in ene)
                        pbar.set_postfix(E=energies)
                        if best_ene is None or any(
                            map(lambda x, y: x.s < 0.5 * y.s, ene, best_ene)
                        ):
                            best_ene = ene
                            log.info(
                                f'Progress: {step + 1}/{steps}, energy = {energies}'
                            )
                    if workdir:
                        if mode == 'training':
                            # the convention is that chkpt-i contains the step i-1 -> i
                            chkpts.update(
                                step + 1,
                                train_state,
                                stats['E_loc/std'].mean(),
                            )
                        table.row['mol_idxs'] = mol_idxs
                        table.row['E_loc'] = E_loc
                        table.row['E_ewm'] = jnp.array(ewm_energies)
                        psi = gather_electrons_on_one_device(train_state.sampler['psi'])
                        if jnp.isnan(psi.log).any():
                            raise NanError()
                        table.row['sign_psi'] = psi.sign[mol_idxs]
                        table.row['log_psi'] = psi.log[mol_idxs]
                        h5file.flush()
                        if metric_logger:
                            single_device_stats = {
                                'energy/ewm': jnp.array([e.n for e in ene]),
                                'energy/ewm_error': jnp.array([e.s for e in ene]),
                            }
                            metric_logger.update(
                                step, single_device_stats, stats, mol_idxs
                            )
                log.info(f'The {mode} has been completed!')
                return train_state
            except NanError:
                pbar.close()
                log.warn('Restarting due to a NaN...')
                if attempt < max_restarts:
                    init_step, train_state = chkpts.last
        log.warn(
            f'The {mode} has crashed before all steps were completed ({step}/{steps})!'
        )
        raise TrainingCrash(train_state)
    finally:
        if pbar:
            pbar.close()
        if workdir:
            chkpts.close()
            metric_logger.close()
            h5file.close()
