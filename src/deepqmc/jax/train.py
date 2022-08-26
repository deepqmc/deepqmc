import logging
import operator
import pickle
from collections import namedtuple
from pathlib import Path

import haiku as hk
import jax
import numpy as np
import tensorboard.summary
from tqdm.auto import tqdm
from uncertainties import ufloat

from .ewm import ewm
from .fit import fit_wf
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
    ewm_state = ewm()
    rng = jax.random.PRNGKey(seed)
    rng, rng_init = jax.random.split(rng)
    params, smpl_state, sample_wf = init_sampling(
        rng_init, hamil, ansatz, state_callback, **(sampling_kwargs or {})
    )
    if workdir:
        chkpts = CheckpointStore(workdir)
        writer = tensorboard.summary.Writer(workdir)

    num_params = jax.tree_util.tree_reduce(
        operator.add, jax.tree_map(lambda x: x.size, params)
    )
    log.info(f'Number of model parameters: {num_params}')

    if steps_eq:
        log.info('Start equilibrating')
        rng, rng_eq = jax.random.split(rng, 2)
        eq_pbar = tqdm(range(steps_eq), desc='equilibrate', disable=None)
        wf = jax.vmap(ansatz.apply, (None, 0, 0))
        for step, rng_eq in zip(eq_pbar, hk.PRNGSequence(rng_eq)):
            smpl_state, eq_stats = equilibration_step(
                rng_eq,
                wf,
                params,
                sample_wf,
                smpl_state,
                state_callback,
            )
            eq_pbar.set_postfix(tau=f'{smpl_state["tau"].item():5.3f}')
            if workdir:
                for k, v in eq_stats.items():
                    writer.add_scalar(f'equilibrate/{k}', v.item(), step)

    log.info('Start training')
    pbar = tqdm(range(steps), desc='train', disable=None)
    enes, best_ene = [], None
    for step, train_state, fit_stats in fit_wf(  # noqa: B007
        rng,
        hamil,
        ansatz,
        params,
        opt,
        sample_wf,
        smpl_state,
        pbar,
        **kwargs,
    ):
        stats = {
            **fit_stats,
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
    return train_state, np.array(enes)


def equilibration_step(rng, wf, params, sample_wf, smpl_state, state_callback):
    _, new_smpl_state, smpl_stats = sample_wf(rng, params, smpl_state)
    if state_callback:
        state, overflow = state_callback(new_smpl_state['wf_state'])
        if overflow:
            smpl_state['wf_state'] = state
            _, new_smpl_state, smpl_stats = sample_wf(rng, params, smpl_state)
    smpl_state = new_smpl_state

    return smpl_state, smpl_stats


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
