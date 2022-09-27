import logging

import haiku as hk

__all__ = ['equilibrate']

log = logging.getLogger(__name__)


def equilibrate(
    rng,
    ansatz,
    state_callback,
    sample_wf,
    params,
    smpl_state,
    pbar,
):
    log.info('Start equilibrating')
    for step, rng in zip(pbar, hk.PRNGSequence(rng)):
        _, new_smpl_state, smpl_stats = sample_wf(rng, params, smpl_state)
        if state_callback:
            state, overflow = state_callback(new_smpl_state['wf_state'])
            if overflow:
                smpl_state['wf_state'] = state
                _, new_smpl_state, smpl_stats = sample_wf(rng, params, smpl_state)
        smpl_state = new_smpl_state

        pbar.set_postfix(tau=f'{smpl_state["tau"].item():5.3f}')
        yield step, smpl_state, smpl_stats
