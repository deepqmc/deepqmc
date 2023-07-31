import logging
import operator
import os
from functools import partial
from itertools import count

import h5py
import jax
import jax.numpy as jnp
import kfac_jax
import optax
from jax import tree_util
from tqdm.auto import tqdm, trange
from uncertainties import ufloat

from .ewm import init_ewm
from .fit import fit_wf
from .log import CheckpointStore, H5LogTable, TensorboardMetricLogger
from .physics import pairwise_self_distance
from .pretrain import pretrain
from .sampling import MultimoleculeSampler, chain, equilibrate
from .utils import (
    ConstantSchedule,
    InverseSchedule,
    gather_on_one_device,
    replicate_on_devices,
    segment_nanmean,
    select_one_device,
    split_on_devices,
    split_rng_key_to_devices,
)
from .wf.base import init_wf_params

__all__ = ['train']

log = logging.getLogger(__name__)

OPT_KWARGS = {
    'adam': {'learning_rate': 1.0e-3, 'b1': 0.9, 'b2': 0.9},
    'adamw': {'learning_rate': 1.0e-3, 'b1': 0.9, 'b2': 0.9},
    'kfac': {
        'learning_rate_schedule': InverseSchedule(0.05, 10000),
        'damping_schedule': ConstantSchedule(0.001),
        'norm_constraint': 0.001,
    },
}


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
    sample_size,
    seed,
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
    mol_idx_factory=None,
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
                :data:`optax` optimzier name). Arguments to the optimizer can be
                passed in :data:`opt_kwargs`.
            - :data:`None`: no optimizer is used, e.g. the evaluation of the Ansatz
                is performed.
        sampler (~deepqmc.sampling.Sampler): a sampler instance
        steps (int): number of optimization steps.
        sample_size (int): the number of samples considered in a batch
        seed (int): the seed used for PRNG.
        mols (Sequence(~deepqmc.molecule.Molecule)): optional, a sequence of molecules
            to consider for transferable training. If None the default molecule from
            hamil is used.
        workdir (str): optional, path, where results should be saved.
        train_state (~deepqmc.fit.TrainState): optional, training checkpoint to
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
        mol_idx_factory (Callable): optional, callback for computing the indices
            of the molecule from which samples are to be taken in a given step.
    """
    assert not sample_size % jax.device_count()
    rng = jax.random.PRNGKey(seed)
    mode = 'evaluation' if opt is None else 'training'
    mols = mols or hamil.mol
    sampler = MultimoleculeSampler(sampler, mols, mol_idx_factory)
    if pretrain_sampler is None:
        pretrain_sampler = sampler
    else:
        pretrain_sampler = chain(*pretrain_sampler[:-1], pretrain_sampler[-1](hamil))
        pretrain_sampler = MultimoleculeSampler(pretrain_sampler, mols, mol_idx_factory)
    if isinstance(opt, str):
        opt_kwargs = OPT_KWARGS.get(opt, {}) | (opt_kwargs or {})
        opt = (
            partial(kfac_jax.Optimizer, **opt_kwargs)
            if opt == 'kfac'
            else getattr(optax, opt)(**opt_kwargs)
        )
    if workdir:
        workdir = os.path.join(workdir, mode)
        chkptdir = os.path.join(chkptdir, mode) if chkptdir else workdir
        os.makedirs(workdir, exist_ok=True)
        os.makedirs(chkptdir, exist_ok=True)
        chkpts = CheckpointStore(chkptdir, **(chkpts_kwargs or {}))
        if workdir:
            metric_logger = (metric_logger or TensorboardMetricLogger)(
                workdir, len(sampler)
            )
        else:
            metric_logger = None
        log.debug('Setting up HDF5 file...')
        h5file = h5py.File(os.path.join(workdir, 'result.h5'), 'a', libver='v110')
        h5file.swmr_mode = True
        tables = []
        for i, mol in enumerate(sampler.mols):
            group = h5file.require_group(str(i)) if len(sampler.mols) > 1 else h5file
            group.attrs.create('geometry', mol.coords.tolist())
            table = H5LogTable(group)
            table.resize(init_step)
            tables.append(table)
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
            num_params = tree_util.tree_reduce(
                operator.add, tree_util.tree_map(lambda x: x.size, params)
            )
            log.info(f'Number of model parameters: {num_params}')
            params = replicate_on_devices(params)
            if pretrain_steps and mode == 'training':
                log.info('Pretraining wrt. baseline wave function')
                rng, rng_pretrain = jax.random.split(rng)
                pretrain_kwargs = pretrain_kwargs or {}
                opt_pretrain = pretrain_kwargs.pop('opt', 'adamw')
                opt_pretrain_kwargs = OPT_KWARGS.get(
                    opt_pretrain, {}
                ) | pretrain_kwargs.pop('opt_kwargs', {})
                if isinstance(opt_pretrain, str):
                    if opt_pretrain == 'kfac':
                        raise NotImplementedError
                    opt_pretrain = getattr(optax, opt_pretrain)
                opt_pretrain = opt_pretrain(**opt_pretrain_kwargs)
                ewm_state, update_ewm = init_ewm(decay_alpha=1.0)
                ewm_states = len(pretrain_sampler) * [ewm_state]
                pbar = tqdm(range(pretrain_steps), desc='pretrain', disable=None)
                for step, params, losses in pretrain(  # noqa: B007
                    rng_pretrain,
                    hamil,
                    ansatz,
                    params,
                    opt_pretrain,
                    pretrain_sampler,
                    steps=pbar,
                    sample_size=sample_size,
                    baseline_kwargs=pretrain_kwargs.pop('baseline_kwargs', {}),
                ):
                    mol_idx = pretrain_sampler.mol_idx(sample_size, step)
                    per_mol_losses = segment_nanmean(
                        losses, mol_idx, len(pretrain_sampler)
                    )
                    ewm_states = [
                        ewm_state if jnp.isnan(loss) else update_ewm(loss, ewm_state)
                        for loss, ewm_state in zip(per_mol_losses, ewm_states)
                    ]
                    ewm_means = [
                        ewm_state.mean if ewm_state.mean is not None else 0.0
                        for ewm_state in ewm_states
                    ]
                    mse_rep = '|'.join(f'{mean:0.2e}' for mean in ewm_means)
                    pbar.set_postfix(MSE=mse_rep)
                    pretrain_stats = {
                        'per_mol': {
                            'MSE': per_mol_losses,
                            'MSE/ewm': jnp.array(
                                [ewm_state.mean for ewm_state in ewm_states]
                            ),
                        }
                    }
                    if metric_logger:
                        metric_logger.update(
                            step, {}, pretrain_stats, prefix='pretraining'
                        )
                log.info(f'Pretraining completed with MSE = {mse_rep}')

        if not train_state or train_state[0] is None:
            rng, rng_eq, rng_smpl_init = split_on_devices(
                split_rng_key_to_devices(rng), 3
            )
            wf = partial(ansatz.apply, select_one_device(params))
            sample_initializer = partial(
                sampler.init, wf=wf, n=sample_size // jax.device_count()
            )
            smpl_state = jax.pmap(sample_initializer)(rng_smpl_init)
            log.info('Equilibrating sampler...')
            pbar = tqdm(
                count() if max_eq_steps is None else range(max_eq_steps),
                desc='equilibrate sampler',
                disable=None,
            )
            for step, smpl_state, smpl_stats in equilibrate(  # noqa: B007
                rng_eq,
                wf,
                sampler,
                smpl_state,
                lambda phys_conf: pairwise_self_distance(phys_conf.r).mean(),
                pbar,
                sample_size,
                block_size=10,
            ):
                tau_rep = '|'.join(
                    f'{tau.mean():.3f}'
                    for tau in sampler.get_state('tau', smpl_state, None)
                )
                pbar.set_postfix(tau=tau_rep)
                if metric_logger:
                    metric_logger.update(step, smpl_stats, prefix='equilibration')
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
                for step, train_state, E_loc, stats in fit_wf(  # noqa: B007
                    rng,
                    hamil,
                    ansatz,
                    opt,
                    sampler,
                    sample_size,
                    pbar,
                    train_state,
                    **(fit_kwargs or {}),
                ):
                    mol_idx = sampler.mol_idx(sample_size, step)
                    per_mol_energy = segment_nanmean(E_loc, mol_idx, len(sampler))
                    ewm_states = [
                        ewm_state if jnp.isnan(ene) else update_ewm(ene, ewm_state)
                        for ene, ewm_state in zip(per_mol_energy, ewm_states)
                    ]
                    ene = [
                        ufloat(e, s)
                        for e, s in zip(
                            [ewm_state.mean for ewm_state in ewm_states],
                            [jnp.sqrt(ewm_state.sqerr) for ewm_state in ewm_states],
                        )
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
                                stats['per_mol']['E_loc/std'].mean(),
                            )
                        for i, (table, ewm_state, smpl_state) in enumerate(
                            zip(tables, ewm_states, train_state.sampler)
                        ):
                            table.row['E_loc'] = E_loc[mol_idx == i]
                            if ewm_state.mean is not None:
                                table.row['E_ewm'] = ewm_state.mean
                            psi = gather_on_one_device(
                                smpl_state['psi'], flatten_device_axis=True
                            )
                            if jnp.isnan(psi.log).any():
                                raise NanError()
                            table.row['sign_psi'] = psi.sign
                            table.row['log_psi'] = psi.log
                        h5file.flush()
                        if metric_logger:
                            single_device_stats = {
                                'per_mol': {
                                    'energy/ewm': jnp.array([e.n for e in ene]),
                                    'energy/ewm_error': jnp.array([e.s for e in ene]),
                                }
                            }
                            metric_logger.update(step, stats, single_device_stats)
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
