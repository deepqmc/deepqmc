import logging

import haiku as hk
from tqdm.auto import tqdm

__all__ = ['equilibrate']

log = logging.getLogger(__name__)


def equilibrate(
    rng,
    ansatz,
    sample_wf,
    params,
    smpl_state,
    *,
    steps,
    writer=None,
    state_callback,
):
    log.info('Start equilibrating')
    pbar = tqdm(range(steps), desc='equilibrate', disable=None)
    for step, rng in zip(pbar, hk.PRNGSequence(rng)):
        _, new_smpl_state, stats = sample_wf(rng, params, smpl_state)
        if state_callback:
            state, overflow = state_callback(new_smpl_state['wf_state'])
            if overflow:
                smpl_state['wf_state'] = state
                _, new_smpl_state, smpl_stats = sample_wf(rng, params, smpl_state)
        smpl_state = new_smpl_state

        pbar.set_postfix(tau=f'{smpl_state["tau"].item():5.3f}')
        if writer:
            for k, v in stats.items():
                writer.add_scalar(f'equilibrate/{k}', v.item(), step)
    return smpl_state
