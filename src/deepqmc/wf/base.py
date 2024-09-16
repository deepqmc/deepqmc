import logging
import operator
from typing import Optional

import jax

from ..optimizer import merge_states
from ..parallel import replicate_on_devices
from ..utils import filter_dict, tree_stack

__all__ = ()

log = logging.getLogger(__name__)


def init_wf_params(
    rng, hamil, ansatz, electronic_states=1, *, merge_keys: Optional[list[str]] = None
):
    rng_sample, *rng_params = jax.random.split(rng, electronic_states + 1)
    phys_conf = hamil.init_sample(rng_sample, hamil.mol.coords, 1)[0]
    params = tree_stack([ansatz.init(rng, phys_conf) for rng in rng_params])
    num_params = jax.tree_util.tree_reduce(
        operator.add, jax.tree_util.tree_map(lambda x: x.size, params)
    )
    state_mult = '' if electronic_states == 1 else f'{electronic_states} x '
    log.info(
        f'Number of model parameters: {state_mult}{num_params // electronic_states}'
    )
    if merge_keys is not None and electronic_states > 1:
        params = merge_states(params, merge_keys)
        merged_params = '\n  - '.join(
            str(key) for key in filter_dict(params, merge_keys).keys()
        )
        log.debug(
            'The following model parameters are shared between the'
            f' {electronic_states} states:\n  - {merged_params}'
        )
    params = replicate_on_devices(params, globally=True)
    return params
