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


def InverseLR(lr, decay_rate):
    return lambda n: lr / (1 + n / decay_rate)


def train(
    hamil,
    ansatz,
    opt,
    workdir=None,
    *,
    sampling_kwargs=None,
    steps,
    seed,
    steps_eq=500,
    state_callback=None,
    **kwargs,
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
        if steps_eq:
            log.info('Equilibrating...')
            rng, rng_eq = jax.random.split(rng)
            pbar = tqdm(range(steps_eq), desc='equilibrate', disable=None)
            for step, smpl_state, smpl_stats in equilibrate(
                rng_eq,
                ansatz,
                sample_wf,
                params,
                smpl_state,
                pbar,
                state_callback=state_callback,
            ):
                if workdir:
                    for k, v in smpl_stats.items():
                        writer.add_scalar(k, v, step - steps_eq)
            pbar.close()
        log.info('Start training')
        pbar = tqdm(range(steps), desc='train', disable=None)
        for step, train_state, fit_stats in fit_wf(  # noqa: B007
            rng,
            hamil,
            ansatz,
            params,
            opt,
            sample_wf,
            smpl_state,
            pbar,
            log_dict=table.row if workdir else None,
            **kwargs,
        ):
            ene = ufloat(fit_stats['energy/ewm'], fit_stats['energy/ewm_error'])
            if ene.s:
                pbar.set_postfix(E=f'{ene:S}')
                log.info(f'Progress: {step + 1}/{steps}, energy = {ene:S}')
            if workdir:
                chkpts.update(fit_stats['E_loc/std'], train_state)
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

    def __init__(self, workdir, size=3, min_interval=100, threshold=0.95):
        self.workdir = Path(workdir)
        for p in self.workdir.glob(self.PATTERN.format('*')):
            p.unlink()
        self.size = size
        self.min_interval = min_interval
        self.threshold = threshold
        self.chkpts = []
        self.step = 0

    def update(self, loss, state) -> None:
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
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        while len(self.chkpts) > self.size:
            self.chkpts.pop(0).path.unlink()
