import logging
import operator
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
from .sampling import MultimoleculeSampler, equilibrate
from .utils import InverseSchedule, segment_nanmean
from .wf.base import init_wf_params, state_callback

__all__ = ['train']

log = logging.getLogger(__name__)

OPT_KWARGS = {
    'adam': {'learning_rate': 1.0e-3, 'b1': 0.9, 'b2': 0.9},
    'adamw': {'learning_rate': 1.0e-3, 'b1': 0.9, 'b2': 0.9},
    'kfac': {
        'learning_rate_schedule': InverseSchedule(0.01, 5000),
        'damping_schedule': InverseSchedule(0.001, 5000),
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
    mols,
    init_sample=None,
    workdir=None,
    train_state=None,
    init_step=0,
    state_callback=state_callback,
    *,
    steps,
    sample_size,
    seed,
    max_restarts=3,
    max_eq_steps=1000,
    pretrain_steps=None,
    pretrain_kwargs=None,
    opt_kwargs=None,
    fit_kwargs=None,
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
        mols (~deepqmc.molecule.Molecule): a molecule or a sequence of molecules to
            consider.
        init_sample (Callable): optional, a method for generating initial samples.
            If `None`, the Hamiltonian's builtin method is used.
        workdir (str): optional, path, where results and checkpoints should be saved.
        train_state (~deepqmc.fit.TrainState): optional, training checkpoint to
            restore training or run evaluation.
        init_step (int): optional, initial step index, useful if
            calculation is restarted from checkpoint saved on disk.
        state_callback (Callable): optional, a function processing the :class:`haiku`
            state of the wave function Ansatz.
        steps (int): optional, number of optimization steps.
        sample_size (int): the number of samples considered in a batch
        seed (int): the seed used for PRNG.
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
        chkpts_kwargs (dict): optional, extra arguments for checkpointing.
        metric_logger: optional, an object that consumes metric logging information.
            If not specified, the default `~.log.TensorboardMetricLogger` is used
            to create tensorboard logs.
        mol_idx_factory (Callable): optional, callback for computing the indices
            of the molecule from which samples are to be taken in a given step.
    """

    rng = jax.random.PRNGKey(seed)
    mode = 'evaluation' if opt is None else 'training'
    sampler = MultimoleculeSampler(sampler, mols, mol_idx_factory)
    if isinstance(opt, str):
        opt_kwargs = OPT_KWARGS.get(opt, {}) | (opt_kwargs or {})
        opt = (
            partial(kfac_jax.Optimizer, **opt_kwargs)
            if opt == 'kfac'
            else getattr(optax, opt)(**opt_kwargs)
        )
    if workdir:
        workdir = f'{workdir}/{mode}'
        chkpts = CheckpointStore(workdir, **(chkpts_kwargs or {}))
        if metric_logger is None and workdir:
            metric_logger = TensorboardMetricLogger(workdir, len(sampler))
        log.debug('Setting up HDF5 file...')
        h5file = h5py.File(f'{workdir}/result.h5', 'a', libver='v110')
        h5file.swmr_mode = True
        tables = []
        for i, mol in enumerate(sampler.mols):
            group = h5file.require_group(str(i))
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
        else:
            rng, rng_init, rng_eq = jax.random.split(rng, 3)
            params = init_wf_params(rng_init, hamil, ansatz, init_sample)
            num_params = tree_util.tree_reduce(
                operator.add, tree_util.tree_map(lambda x: x.size, params)
            )
            log.info(f'Number of model parameters: {num_params}')
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
                ewm_states = len(sampler) * [ewm_state]
                pbar = tqdm(range(pretrain_steps), desc='pretrain', disable=None)
                for step, params, losses in pretrain(  # noqa: B007
                    rng_pretrain,
                    hamil,
                    ansatz,
                    opt_pretrain,
                    sampler,
                    init_sample=init_sample,
                    steps=pbar,
                    sample_size=sample_size,
                    baseline_kwargs=pretrain_kwargs.pop('baseline_kwargs', {}),
                ):
                    mol_idx = sampler.mol_idx(sample_size, step)
                    per_mol_losses = segment_nanmean(losses, mol_idx, len(sampler))
                    ewm_states = [
                        ewm_state if jnp.isnan(loss) else update_ewm(loss, ewm_state)
                        for loss, ewm_state in zip(per_mol_losses, ewm_states)
                    ]
                    ewm_means = [ewm_state.mean if ewm_state.mean is not None else 0.0 for ewm_state in ewm_states]
                    mse_rep = '|'.join(
                        f'{mean:0.2e}' for mean in ewm_means
                    )
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
                        metric_logger.update(step, pretrain_stats, prefix='pretraining')
                log.info(f'Pretraining completed with MSE = {mse_rep}')
            smpl_state = sampler.init(
                rng, partial(ansatz.apply, params), sample_size, state_callback
            )
            log.info('Equilibrating sampler...')
            pbar = tqdm(
                count() if max_eq_steps is None else range(max_eq_steps),
                desc='equilibrate sampler',
                disable=None,
            )
            for step, smpl_state, smpl_stats in equilibrate(  # noqa: B007
                rng_eq,
                partial(ansatz.apply, params),
                sampler,
                smpl_state,
                lambda phys_conf: pairwise_self_distance(phys_conf.r).mean(),
                pbar,
                sample_size,
                state_callback,
                block_size=10,
            ):
                tau_rep = '|'.join(
                    f'{tau:.3f}' for tau in sampler.get_state('tau', smpl_state, None)
                )
                pbar.set_postfix(tau=tau_rep)
                if metric_logger:
                    metric_logger.update(step, smpl_stats, prefix='equilibration')
            pbar.close()
            train_state = smpl_state, params, None
            if workdir and mode == 'training':
                chkpts.dump(init_step, train_state)
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
                    state_callback,
                    train_state,
                    **(fit_kwargs or {}),
                ):
                    psi = sampler.get_state(
                        'psi',
                        train_state.sampler,
                        sampler.select_idxs(sample_size, step),
                    )
                    if jnp.isnan(psi.log).any():
                        raise NanError()
                    mol_idx = sampler.mol_idx(sample_size, step)
                    per_mol_energy = segment_nanmean(E_loc, mol_idx, len(sampler))
                    ewm_states = [
                        ewm_state if jnp.isnan(ene) else update_ewm(ene, ewm_state)
                        for ene, ewm_state in zip(per_mol_energy, ewm_states)
                    ]
                    stats['per_mol'] = {
                        'energy/ewm': jnp.array(
                            [ewm_state.mean for ewm_state in ewm_states]
                        ),
                        'energy/ewm_error': jnp.sqrt(
                            jnp.array([ewm_state.sqerr for ewm_state in ewm_states])
                        ),
                        **stats['per_mol'],
                    }
                    ene = [
                        ufloat(e, s)
                        for e, s in zip(
                            stats['per_mol']['energy/ewm'],
                            stats['per_mol']['energy/ewm_error'],
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
                        for table, E, ewm_state, smpl_state in zip(
                            tables, per_mol_energy, ewm_states, train_state.sampler
                        ):
                            table.row['E_mean'] = E
                            if ewm_state.mean is not None:
                                table.row['E_ewm'] = ewm_state.mean
                            table.row['sign_psi'] = smpl_state['psi'].sign
                            table.row['log_psi'] = smpl_state['psi'].log
                        h5file.flush()
                        if metric_logger:
                            metric_logger.update(step, stats)
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
