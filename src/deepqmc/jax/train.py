import logging
import pickle
from collections import namedtuple
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tensorboard.summary
from tqdm.auto import tqdm
from uncertainties import ufloat

from .ewm import ewm
from .fit import fit_wf
from .sampling import DecorrSampler

__all__ = 'train'

log = logging.getLogger(__name__)


def InverseLR(lr, decay_rate):
    return lambda n: lr / (1 + n / decay_rate)


def train(
    hamil,
    ansatz,
    opt,
    sampler,
    edge_builder,
    workdir=None,
    *,
    steps,
    sample_size,
    decorr,
    seed,
    equilibrate=500,
    **params,
):
    ewm_state = ewm()
    sampler = sampler(hamil)
    sampler = DecorrSampler(sampler, decorr=decorr)
    rng = jax.random.PRNGKey(seed)
    if workdir:
        chkpts = CheckpointStore(workdir)
        writer = tensorboard.summary.Writer(workdir)
    log.info('Start training')
    eq_pbar = (
        tqdm(range(equilibrate), desc='equilibrate', disable=None)
        if equilibrate
        else None
    )
    pbar = tqdm(range(steps), desc='train', disable=None)
    enes, best_ene = [], None
    for step, train_state, E_loc in fit_wf(  # noqa: B007
        rng,
        hamil,
        ansatz,
        opt,
        sampler,
        edge_builder,
        sample_size,
        pbar,
        eq_pbar,
        **params,
    ):
        stats = {
            'E_loc/mean': jnp.mean(E_loc).item(),
            'E_loc/std': jnp.std(E_loc).item(),
            'E_loc/max': jnp.max(E_loc).item(),
            'E_loc/min': jnp.min(E_loc).item(),
            **hamil.stats(train_state.sampler['r']),
        }
        ewm_state = ewm(stats['E_loc/mean'], ewm_state)
        ene = ufloat(ewm_state.mean, np.sqrt(ewm_state.sqerr))
        enes.append(ene)
        if ene.s:
            pbar.set_postfix(E=f'{ene:S}')
            if best_ene is None or ene.n < best_ene.n - 3 * ene.s:
                best_ene = ene
                log.info(f'Progress: {step + 1}/{steps}, energy = {ene:S}')
        if workdir:
            chkpts.update(stats['E_loc/std'], train_state)
            writer.add_scalar('energy/ewm', ene.n, step)
            for k, v in stats.items():
                writer.add_scalar(k, v, step)
    return train_state, E_loc, np.array(enes)


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
