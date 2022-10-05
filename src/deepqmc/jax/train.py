import logging
import operator
import pickle
from collections import namedtuple
from pathlib import Path

import h5py
import jax
import tensorboard.summary
from tqdm.auto import tqdm
from uncertainties import ufloat

from .equilibrate import equilibrate
from .fit import fit_wf
from .log import H5LogTable
from .sampling import init_sampling

__all__ = 'train'

log = logging.getLogger(__name__)


def train(
    hamil,
    ansatz,
    state_callback,
    opt,
    seed,
    workdir=None,
    save_every=None,
    *,
    n_steps=1e4,
    n_steps_eq=500,
    sampling_kwargs=None,
    fit_kwargs=None,
):
    rng = jax.random.PRNGKey(seed)
    rng, rng_init = jax.random.split(rng)
    params, smpl_state, sample_wf = init_sampling(
        rng_init, hamil, ansatz, state_callback, **(sampling_kwargs or {})
    )
    if workdir:
        chkpts = CheckpointStore(workdir)
        writer = tensorboard.summary.Writer(workdir)
        log.debug('Setting up HDF5 file...')
        h5file = h5py.File(f'{workdir}/fit.h5', 'a', libver='v110')
        h5file.swmr_mode = True
        table = H5LogTable(h5file)
        h5file.flush()
        log.debug('Done')

    num_params = jax.tree_util.tree_reduce(
        operator.add, jax.tree_map(lambda x: x.size, params)
    )
    log.info(f'Number of model parameters: {num_params}')
    try:
        if n_steps_eq:
            log.info('Equilibrating...')
            rng, rng_eq = jax.random.split(rng)
            pbar = tqdm(range(n_steps_eq), desc='equilibrate', disable=None)
            for step, smpl_state, smpl_stats in equilibrate(  # noqa: B007
                rng_eq,
                ansatz,
                state_callback,
                sample_wf,
                params,
                smpl_state,
                pbar,
            ):
                if workdir:
                    for k, v in smpl_stats.items():
                        writer.add_scalar(k, v, step - n_steps_eq)
            pbar.close()
        log.info('Start training')
        pbar = tqdm(range(n_steps), desc='train', disable=None)
        for step, train_state, fit_stats in fit_wf(  # noqa: B007
            rng,
            hamil,
            ansatz,
            state_callback,
            params,
            opt,
            sample_wf,
            smpl_state,
            pbar,
            log_dict=table.row if workdir else None,
            **(fit_kwargs or {}),
        ):
            ene = ufloat(fit_stats['energy/ewm'], fit_stats['energy/ewm_error'])
            if ene.s:
                pbar.set_postfix(E=f'{ene:S}')
                log.info(f'Progress: {step + 1}/{n_steps}, energy = {ene:S}')
            if workdir:
                if save_every and not (step + 1) % save_every:
                    chkpts.store(step + 1, train_state)
                for k, v in fit_stats.items():
                    writer.add_scalar(k, v, step)
        return train_state
    finally:
        pbar.close()
        if workdir:
            writer.close()
            h5file.close()


Checkpoint = namedtuple('Checkpoint', 'step loss path')


class CheckpointStore:
    PATTERN = 'chkpt-{}.pt'

    def __init__(self, workdir):
        self.workdir = Path(workdir)
        for p in self.workdir.glob(self.PATTERN.format('*')):
            p.unlink()

    def store(self, step, state) -> None:
        path = self.workdir / self.PATTERN.format(step)
        with open(path, 'wb') as f:
            pickle.dump(state, f)
