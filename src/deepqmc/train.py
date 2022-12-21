import logging
import operator
from functools import partial
from itertools import count

import h5py
import jax
import jax.numpy as jnp
import kfac_jax
import optax
import tensorboard.summary
from tqdm.auto import tqdm, trange
from uncertainties import ufloat

from .ewm import init_ewm
from .fit import fit_wf
from .log import CheckpointStore, H5LogTable, update_tensorboard_writer
from .physics import pairwise_self_distance
from .pretrain import pretrain
from .sampling import equilibrate
from .utils import InverseSchedule
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
    """

    rng = jax.random.PRNGKey(seed)
    mode = 'evaluation' if opt is None else 'training'
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
        writer = tensorboard.summary.Writer(workdir)
        log.debug('Setting up HDF5 file...')
        h5file = h5py.File(f'{workdir}/result.h5', 'a', libver='v110')
        h5file.swmr_mode = True
        table = H5LogTable(h5file)
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
            params = init_wf_params(rng_init, hamil, ansatz)
            num_params = jax.tree_util.tree_reduce(
                operator.add, jax.tree_map(lambda x: x.size, params)
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
                ewm_state, update_ewm = init_ewm(decay_alpha=1)
                pbar = tqdm(range(pretrain_steps), desc='pretrain', disable=None)
                for step, params, loss in pretrain(  # noqa: B007
                    rng_pretrain,
                    hamil,
                    ansatz,
                    opt_pretrain,
                    sampler,
                    steps=pbar,
                    sample_size=sample_size,
                    baseline_kwargs=pretrain_kwargs.pop('baseline_kwargs', {}),
                ):
                    ewm_state = update_ewm(loss.item(), ewm_state)
                    pbar.set_postfix(MSE=f'{ewm_state.mean:0.2e}')
                    pretrain_stats = {'MSE': loss.item(), 'MSE/ewm': ewm_state.mean}
                    if workdir:
                        update_tensorboard_writer(
                            writer, step, pretrain_stats, prefix='pretraining'
                        )
                log.info(f'Pretraining completed with MSE = {ewm_state.mean:0.2e}')
            smpl_state = sampler.init(
                rng, partial(ansatz.apply, params), sample_size, state_callback
            )
            log.info('Equilibrating sampler...')
            pbar = tqdm(
                count() if max_eq_steps is None else range(max_eq_steps),
                desc='equilibrate',
                disable=None,
            )
            for step, smpl_state, smpl_stats in equilibrate(  # noqa: B007
                rng_eq,
                partial(ansatz.apply, params),
                sampler,
                smpl_state,
                lambda r: pairwise_self_distance(r).mean(),
                pbar,
                state_callback,
                block_size=10,
            ):
                pbar.set_postfix(tau=f'{smpl_state["tau"].item():5.3f}')
                if workdir:
                    update_tensorboard_writer(
                        writer, step, smpl_stats, prefix='equilibration'
                    )
            pbar.close()
            train_state = smpl_state, params, None
            if workdir and mode == 'training':
                chkpts.dump(init_step, train_state)
            log.info(f'Start {mode}')
        best_ene = None
        ewm_state, update_ewm = init_ewm()
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
                    if jnp.isnan(train_state.sampler['psi'].log).any():
                        raise NanError()
                    ewm_state = update_ewm(stats['E_loc/mean'], ewm_state)
                    stats = {
                        'energy/ewm': ewm_state.mean,
                        'energy/ewm_error': jnp.sqrt(ewm_state.sqerr),
                        **stats,
                    }
                    ene = ufloat(stats['energy/ewm'], stats['energy/ewm_error'])
                    if ene.s:
                        pbar.set_postfix(E=f'{ene:S}')
                        if best_ene is None or ene.s < 0.5 * best_ene.s:
                            best_ene = ene
                            log.info(f'Progress: {step + 1}/{steps}, energy = {ene:S}')
                    if workdir:
                        if mode == 'training':
                            # the convention is that chkpt-i contains the step i-1 -> i
                            chkpts.update(step + 1, train_state, stats['E_loc/std'])
                        table.row['E_loc'] = E_loc
                        table.row['E_ewm'] = ewm_state.mean
                        table.row['sign_psi'] = train_state.sampler['psi'].sign
                        table.row['log_psi'] = train_state.sampler['psi'].log
                        h5file.flush()
                        update_tensorboard_writer(writer, step, stats)
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
            writer.close()
            h5file.close()
