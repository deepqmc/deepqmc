import logging
import operator
import pickle
from collections import namedtuple
from copy import deepcopy
from functools import partial
from itertools import count
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import kfac_jax
import optax
import tensorboard.summary
from tqdm.auto import tqdm, trange
from uncertainties import ufloat

from .ewm import ewm
from .fit import fit_wf, init_fit
from .log import H5LogTable, update_tensorboard_writer
from .physics import pairwise_self_distance
from .pretrain import pretrain
from .sampling import equilibrate
from .utils import InverseSchedule
from .wf.base import state_callback

__all__ = ['train']

log = logging.getLogger(__name__)


OPT_KWARGS = {
    'adam': {'lr': 1.0e-3, 'b1': 0.9, 'b2': 0.9},
    'adamw': {'lr': 1.0e-3, 'b1': 0.9, 'b2': 0.9},
    'kfac': {
        'learning_rate_schedule': InverseSchedule(0.01, 5000),
        'damping_schedule': InverseSchedule(0.001, 5000),
        'norm_constraint': 0.001,
    },
}


class NanError(Exception):
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
    state_callback=state_callback,
    *,
    steps,
    sample_size,
    seed,
    init_step=0,
    max_restarts=3,
    pretrain_steps=None,
    pretrain_baseline_kwargs=None,
    **kwargs,
):
    r"""Train or evaluate a JAX wave function model.

    It initializes and equilibrates the MCMC sampling of the wave function ansatz,
    then optimizes or samples it using the variational principle. It optionally
    saves checkpoints and rewinds the training/evaluation if an error is encountered.
    If an optimizer is supplied, the Ansatz is optimized, otherwise the Ansatz is
    only sampled.

    Args:
        hamil (~jax.hamil.Hamiltonian): the Hamiltonian of the physical system.
        ansatz (~jax.wf.WaveFunction): the wave function ansatz.
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

        sampler (~jax.sampling.Sampler): a sampler instance
        workdir (str): optional, path, where results and checkpoints should be saved.
        state_callback (Callable): optional, a function processing the :class:`haiku`
            state of the wave function ansatz.
        steps (int): optional, number of optimization steps.
        sample_size (int): the number of samples considered in a batch
        seed (int): the seed used for PRNG.
        init_step (int): optional, initial step index, useful if
        calculation is restarted from checkpoint saved on disk.
        max_restarts (int): optional, the maximum number of times the training is
            retried before a :class:`NaNError` is raised.
        pretrain_steps (int): optional, the number of pretraining steps wrt. to the
            Baseline wave function obtained with pyscf.
        pretrain_baseline_kwargs (dict): optional, arguments passed to the baseline
            wave function.
        kwargs (dict): optional, extra arguments passed to the :func:`~.fit.fit_wf`
            function.
    """

    ewm_state = ewm()
    rng = jax.random.PRNGKey(seed)
    mode = 'evaluate' if opt is None else 'train'
    if isinstance(opt, str):
        opt_kwargs = kwargs.pop('opt_kwargs', OPT_KWARGS)[opt]
        opt = (
            partial(kfac_jax.Optimizer, **opt_kwargs)
            if opt == 'kfac'
            else getattr(optax, opt)(**opt_kwargs)
        )
    if workdir:
        workdir = f'{workdir}/{mode}'
        chkpts = CheckpointStore(workdir, **kwargs.pop('chkpts_kwargs', {}))
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
                    'train': f'Restart training from step {init_step}',
                    'evaluate': 'Start evaluate',
                }[mode]
            )
            kwargs.pop('max_eq_steps', None)
        else:
            params, smpl_state = init_fit(
                rng, hamil, ansatz, sampler, sample_size, state_callback
            )
            num_params = jax.tree_util.tree_reduce(
                operator.add, jax.tree_map(lambda x: x.size, params)
            )
            log.info(f'Number of model parameters: {num_params}')
            if pretrain_steps and mode == 'train':
                log.info('Pretraining wrt. HF wave function')
                pbar = tqdm(range(pretrain_steps), desc='pretrain', disable=None)
                for step, params, loss, pretrain_stats in pretrain(  # noqa: B007
                    rng,
                    hamil,
                    ansatz,
                    opt,
                    sampler,
                    steps=pbar,
                    sample_size=sample_size,
                    baseline_kwargs=pretrain_baseline_kwargs,
                ):
                    pbar.set_postfix(MSE=f'{loss.item():0.5e}')
                    pretrain_stats = {
                        'pretraining/MSE': loss.item(),
                        **pretrain_stats,
                    }
                    if workdir:
                        update_tensorboard_writer(writer, step, pretrain_stats)
            log.info('Equilibrating sampler...')
            pbar = tqdm(count(), desc='equilibrate', disable=None)
            for _, smpl_state, smpl_stats in equilibrate(  # noqa: B007
                rng,
                partial(ansatz.apply, params),
                sampler,
                smpl_state,
                lambda r: pairwise_self_distance(r).mean(),
                pbar,
                state_callback,
                block_size=10,
                max_steps=kwargs.pop('max_eq_steps', None),
            ):
                pbar.set_postfix(tau=f'{smpl_state["tau"].item():5.3f}')
                # TODO
                # if workdir:
                #     update_tensorboard_writer(writer, step, stats)
            pbar.close()
            train_state = smpl_state, params, None
            if workdir and mode == 'train':
                chkpts.dump(train_state)
            log.info(f'Start {mode}')
        pbar = trange(
            init_step, steps, initial=init_step, total=steps, desc=mode, disable=None
        )
        best_ene = None
        for _ in range(max_restarts):
            nan = False
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
                **kwargs,
            ):
                if jnp.isnan(train_state.sampler['psi'].log).any():
                    log.warn('Restarting due to a NaN...')
                    step, train_state = chkpts.last
                    pbar.close()
                    pbar = trange(
                        step, steps, initial=step, total=steps, desc=mode, disable=None
                    )
                    nan = True
                    break
                ewm_state = ewm(stats['E_loc/mean'], ewm_state)
                stats = {
                    'energy/ewm': ewm_state.mean,
                    'energy/ewm_error': jnp.sqrt(ewm_state.sqerr),
                    **stats,
                }
                ene = ufloat(stats['energy/ewm'], stats['energy/ewm_error'])
                if ene.s:
                    pbar.set_postfix(E=f'{ene:S}')
                    if best_ene is None or ene.n < best_ene.n - 3 * ene.s:
                        best_ene = ene
                        log.info(f'Progress: {step + 1}/{steps}, energy = {ene:S}')
                if workdir:
                    if mode == 'train':
                        chkpts.update(stats['E_loc/std'], train_state)
                    table.row['E_loc'] = E_loc
                    table.row['E_ewm'] = ewm_state.mean
                    table.row['sign_psi'] = train_state.sampler['psi'].sign
                    table.row['log_psi'] = train_state.sampler['psi'].log
                    update_tensorboard_writer(writer, step, stats)
            if not nan:
                return train_state
        raise NanError(train_state)
    finally:
        if pbar:
            pbar.close()
        if workdir:
            chkpts.close()
            writer.close()
            h5file.close()


Checkpoint = namedtuple('Checkpoint', 'step loss path')


class CheckpointStore:
    PATTERN = 'chkpt-{}.pt'

    def __init__(self, workdir, size=3, min_interval=100, threshold=0.95):
        self.workdir = Path(workdir)
        for p in self.workdir.glob(self.PATTERN.format('*')):
            p.unlink()
        self.size = size
        self.min_interval = min_interval
        self.threshold = threshold
        self.chkpts = []
        self.step = 0
        self.buffer = None

    def update(self, loss, state):
        self.step += 1
        self.buffer = deepcopy(state)
        if (
            self.step < self.min_interval
            or self.chkpts
            and (
                self.step < self.min_interval + self.chkpts[-1].step
                or self.threshold
                and loss > self.threshold * self.chkpts[-1].loss
            )
        ):
            return
        path = self.dump(state)
        self.chkpts.append(Checkpoint(self.step, loss, path))
        while len(self.chkpts) > self.size:
            self.chkpts.pop(0).path.unlink()

    def dump(self, state):
        path = self.workdir / self.PATTERN.format(self.step)
        with path.open('wb') as f:
            pickle.dump((self.step, state), f)
        return path

    def close(self):
        if self.buffer is not None:
            self.dump(self.buffer)

    @property
    def last(self):
        chkpt = self.chkpts[-1]
        with chkpt.path.open('rb') as f:
            return pickle.load(f)
