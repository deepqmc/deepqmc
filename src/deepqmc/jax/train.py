import logging
import operator
import pickle
from collections import namedtuple
from functools import partial
from itertools import count
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import tensorboard.summary
from tqdm.auto import tqdm
from uncertainties import ufloat

from .ewm import ewm
from .fit import fit_wf, init_fit
from .log import H5LogTable
from .physics import pairwise_self_distance
from .sampling import equilibrate
from .wf.base import state_callback

__all__ = ['train']

log = logging.getLogger(__name__)


def train(
    hamil,
    ansatz,
    opt,
    sampler,
    workdir=None,
    state_callback=state_callback,
    *,
    steps,
    sample_size,
    seed,
    max_restarts=3,
    **kwargs,
):
    ewm_state = ewm()
    rng = jax.random.PRNGKey(seed)
    if workdir:
        chkpts = CheckpointStore(workdir)
        writer = tensorboard.summary.Writer(workdir)
        log.debug('Setting up HDF5 file...')
        h5file = h5py.File(f'{workdir}/fit.h5', 'a', libver='v110')
        h5file.swmr_mode = True
        table = H5LogTable(h5file)
        h5file.flush()
    pbar = None
    try:
        params, smpl_state = init_fit(
            rng, hamil, ansatz, sampler, sample_size, state_callback
        )
        num_params = jax.tree_util.tree_reduce(
            operator.add, jax.tree_map(lambda x: x.size, params)
        )
        log.info(f'Number of model parameters: {num_params}')
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
        ):
            pbar.set_postfix(tau=f'{smpl_state["tau"].item():5.3f}')
            # TODO
            # if workdir:
            #     for k, v in smpl_stats.items():
            #         writer.add_scalar(k, v, step)
        pbar.close()
        log.info('Start training')
        pbar = tqdm(range(steps), desc='train', disable=None)
        best_ene = None
        train_state = params, None, smpl_state
        for _ in range(max_restarts):
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
                    pbar = tqdm(range(step, steps), desc='train', disable=None)
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
                    chkpts.update(stats['E_loc/std'], train_state)
                    table.row['E_loc'] = E_loc
                    table.row['E_ewm'] = ewm_state.mean
                    table.row['sign_psi'] = train_state.sampler['psi'].sign
                    table.row['log_psi'] = train_state.sampler['psi'].log
                    for k, v in stats.items():
                        writer.add_scalar(k, v, step)
            return train_state
    finally:
        if pbar:
            pbar.close()
        if workdir:
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

    def update(self, loss, state):
        self.step += 1
        if (
            self.step < self.min_interval
            or self.chkpts
            and (
                self.step < self.min_interval + self.chkpts[-1].step
                or loss > self.threshold * self.chkpts[-1].loss
            )
        ):
            return
        path = self.workdir / self.PATTERN.format(self.step)
        self.chkpts.append(Checkpoint(self.step, loss, path))
        with path.open('wb') as f:
            pickle.dump(state, f)
        while len(self.chkpts) > self.size:
            self.chkpts.pop(0).path.unlink()

    @property
    def last(self):
        chkpt = self.chkpts[-1]
        with chkpt.path.open('rb') as f:
            return chkpt.step, pickle.load(f)
