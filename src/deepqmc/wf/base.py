import logging
import operator
from typing import Optional

import jax
import jax.numpy as jnp

from ..optimizer import merge_states
from ..parallel import replicate_on_devices
from ..types import PhysicalConfiguration
from ..utils import filter_dict, tree_stack

__all__ = ()

log = logging.getLogger(__name__)


def init_wf_params(
    rng, hamil, ansatz, electronic_states=1, *, merge_keys: Optional[list[str]] = None
):
    rng_params = jax.random.split(rng, electronic_states)
    dummy_phys_conf = PhysicalConfiguration(  # type: ignore
        jnp.zeros_like(hamil.mol.coords),
        jnp.zeros((hamil.n_up + hamil.n_down, 3)),
        jnp.array(0),
    )
    params = tree_stack([ansatz.init(rng, dummy_phys_conf) for rng in rng_params])
    num_params = jax.tree_util.tree_reduce(
        operator.add, jax.tree_util.tree_map(lambda x: x.size, params)
    )
    state_mult = '' if electronic_states == 1 else f'{electronic_states} x '
    log.info(
        f'Number of model parameters: {state_mult}{num_params // electronic_states}'
    )
    if merge_keys is not None and electronic_states > 1:
        params = merge_states(
            params, tuple(merge_keys) if merge_keys is not None else None
        )
        merged_params = '\n  - '.join(
            str(key) for key in filter_dict(params, merge_keys).keys()
        )
        log.debug(
            'The following model parameters are shared between the'
            f' {electronic_states} states:\n  - {merged_params}'
        )
    params = replicate_on_devices(params, globally=True)
    return params
