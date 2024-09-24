from pathlib import Path

import jax
import jax.numpy as jnp

from ..log import CheckpointStore
from ..types import Params, PhysicalConfiguration


def load_parameters(
    chkpt_path: Path, squeeze_electronic_states: bool = False
) -> Params:
    r"""Load ansatz parameters from a checkpoint file."""
    _step, train_state = CheckpointStore.load(chkpt_path, deserialize=False)
    params = train_state.params
    if squeeze_electronic_states:
        params = jax.tree_util.tree_map(lambda x: x.squeeze(axis=0), params)

    return params


def phys_conf_from_checkpoint(chkpt_path: Path) -> PhysicalConfiguration:
    r"""Load :class:`~deepqmc.types.PhysicalConfiguration`s from a checkpoint file."""
    _step, train_state = CheckpointStore.load(chkpt_path, deserialize=False)

    r = train_state.sampler['elec']['r']
    R = train_state.sampler['nuc']['R'][
        :, None, None
    ]  # Electronic state and electron batch dims
    R = jnp.broadcast_to(R, (*r.shape[:-2], *R.shape[-2:]))

    return PhysicalConfiguration(R, r, jnp.zeros(r.shape[:-2], dtype=int))  # type: ignore
