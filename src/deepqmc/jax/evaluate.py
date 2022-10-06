import logging
from functools import partial

import h5py
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorboard.summary
from tqdm.auto import tqdm
from uncertainties import ufloat

from .ewm import ewm
from .log import H5LogTable
from .physics import pairwise_self_distance
from .sampling import equilibrate, init_sampling

__all__ = ['evaluate']

log = logging.getLogger(__name__)


def evaluate(  # noqa: C901
    hamil,
    ansatz,
    state_callback,
    params,
    steps,
    seed,
    workdir=None,
    sampling_kwargs=None,
    *,
    n_steps_eq=500,
):
    ewm_state = ewm()
    rng = jax.random.PRNGKey(seed)
    rng, rng_init = jax.random.split(rng)
    _, smpl_state, sampler = init_sampling(
        rng_init,
        hamil,
        ansatz,
        state_callback,
        params=params,
        **(sampling_kwargs or {}),
    )
    if workdir:
        writer = tensorboard.summary.Writer(workdir)
        log.debug('Setting up HDF5 file...')
        h5file = h5py.File(f'{workdir}/sample.h5', 'a', libver='v110')
        h5file.swmr_mode = True
        table = H5LogTable(h5file)
        h5file.flush()
        log.debug('Done')

    @jax.jit
    def eval_step(rng, smpl_state):
        rs, smpl_state, smpl_stats = sampler.sample(
            smpl_state, rng, partial(ansatz.apply, params)
        )
        wf = lambda state, rs: ansatz.apply(params, state, rs)[0]
        E_loc, hamil_stats = jax.vmap(hamil.local_energy(wf))(
            smpl_state['wf_state'], rs
        )
        stats = {
            **smpl_stats,
            'E_loc/mean': jnp.mean(E_loc),
            'E_loc/std': jnp.std(E_loc),
            'E_loc/max': jnp.max(E_loc),
            'E_loc/min': jnp.min(E_loc),
            **jax.tree_util.tree_map(jnp.mean, hamil_stats),
        }
        return smpl_state, E_loc, stats

    try:
        if n_steps_eq:
            log.info('Equilibrating...')
            rng, rng_eq = jax.random.split(rng)
            pbar = tqdm(range(n_steps_eq), desc='equilibrate', disable=None)
            for step, smpl_state, smpl_stats in equilibrate(  # noqa: B007
                rng_eq,
                partial(ansatz.apply, params),
                sampler,
                smpl_state,
                lambda r: pairwise_self_distance(r).mean().item(),
                pbar,
                state_callback,
                block_size=10,
            ):
                if workdir:
                    for k, v in smpl_stats.items():
                        writer.add_scalar(k, v, step - n_steps_eq)
            pbar.close()

        log.info('Start evaluating')
        pbar = tqdm(range(steps), desc='evaluate', disable=None)
        enes = []
        best_err = None
        for step, rng in zip(pbar, hk.PRNGSequence(rng)):
            smpl_state, E_loc, stats = eval_step(rng, smpl_state_prev := smpl_state)
            if state_callback:
                wf_state, overflow = state_callback(smpl_state['wf_state'])
                if overflow:
                    smpl_state = smpl_state_prev
                    smpl_state['wf_state'] = wf_state
                    continue
            ewm_state = ewm(stats['E_loc/mean'], ewm_state)
            stats = {
                'energy/ewm': ewm_state.mean,
                'energy/ewm_error': jnp.sqrt(ewm_state.sqerr),
                **stats,
            }
            ene = ufloat(stats['energy/ewm'], stats['energy/ewm_error'])
            enes.append(ene)
            if ene.s:
                pbar.set_postfix(E=f'{ene:S}')
                if best_err is None or ene.s < 0.5 * best_err:
                    best_err = ene.s
                    log.info(f'Progress: {step + 1}/{steps}, energy = {ene:S}')
            if workdir:
                for k, v in stats.items():
                    writer.add_scalar(k, v, step)
                table.row['E_loc'] = E_loc
                table.row['E_ewm'] = ewm_state.mean
                table.row['sign_psi'] = smpl_state['psi'].sign
                table.row['log_psi'] = smpl_state['psi'].log
        return np.array(enes)
    finally:
        pbar.close()
        if workdir:
            writer.close()
            h5file.close()
