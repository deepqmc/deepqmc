import logging

import h5py
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorboard.summary
from tqdm.auto import tqdm
from uncertainties import ufloat

from .equilibrate import equilibrate
from .ewm import ewm
from .log import H5LogTable
from .sampling import init_sampling

__all__ = ['evaluate']

log = logging.getLogger(__name__)


def evaluate(
    hamil,
    ansatz,
    params,
    steps,
    seed,
    workdir=None,
    sampling_kwargs=None,
    *,
    steps_eq=500,
    state_callback=None,
):
    ewm_state = ewm()
    rng = jax.random.PRNGKey(seed)
    rng, rng_init = jax.random.split(rng)
    _, smpl_state, sample_wf = init_sampling(
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

    if steps_eq:
        rng, rng_eq = jax.random.split(rng)
        smpl_state = equilibrate(
            rng_eq,
            ansatz,
            sample_wf,
            params,
            smpl_state,
            steps=steps_eq,
            writer=writer if workdir else None,
            state_callback=state_callback,
        )

    @jax.jit
    def eval_step(rng, smpl_state):
        rs, smpl_state, smpl_stats = sample_wf(rng, params, smpl_state)
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
        return smpl_state, stats, E_loc

    log.info('Start evaluating')
    pbar = tqdm(range(steps), desc='evaluate', disable=None)
    wf = lambda state, rs: ansatz.apply(params, state, rs)[0].log
    enes = []
    try:
        for step, rng in zip(pbar, hk.PRNGSequence(rng)):
            log_dict = table.row if workdir else None
            new_smpl_state, eval_stats, E_loc = eval_step(rng, smpl_state)
            if state_callback:
                state, overflow = state_callback(new_smpl_state['wf_state'])
                if overflow:
                    smpl_state['wf_state'] = state
                    _, new_smpl_state, smpl_stats = sample_wf(rng, smpl_state)
            smpl_state = new_smpl_state

            ewm_state = ewm(eval_stats['E_loc/mean'], ewm_state)
            eval_stats = {
                'energy/ewm': ewm_state.mean,
                'energy/ewm_error': jnp.sqrt(ewm_state.sqerr),
                **eval_stats,
            }
            ene = ufloat(ewm_state.mean, np.sqrt(ewm_state.sqerr))
            enes.append(ene)

            if ene.s:
                pbar.set_postfix(E=f'{ene:S}')
                log.info(f'Progress: {step + 1}/{steps}, energy = {ene:S}')
            if workdir:
                for k, v in eval_stats.items():
                    writer.add_scalar(k, v, step)
            if log_dict:
                log_dict['E_loc'] = E_loc
                log_dict['E_ewm'] = ewm_state.mean
                log_dict['sign_psi'] = smpl_state['psi'].sign
                log_dict['log_psi'] = smpl_state['psi'].log
        return np.array(enes)
    finally:
        pbar.close()
        if workdir:
            writer.close()
            h5file.close()
