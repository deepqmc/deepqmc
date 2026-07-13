from pathlib import Path

import jax
import jax.numpy as jnp
from typing import Optional

from ..log import CheckpointStore
from ..types import Params, PhysicalConfiguration


def load_parameters(
    chkpt_path: Path,
    state: Optional[int] = None,
) -> Params:
    r"""Load ansatz parameters from a checkpoint file.

    Args:
        chkpt_path (~pathlib.Path): path to a ``chkpt-*.pt`` file written by
            :class:`~deepqmc.log.CheckpointStore`.
        state (int, optional): if the ansatz has multiple electronic states, the
            index of the state to load parameters for. If ``None``, the parameters
            for all electronic states are returned, with the electronic-state
            dimension preserved.

    Returns:
        ~deepqmc.types.Params: the ansatz parameters stored in the checkpoint.
    """
    _step, train_state = CheckpointStore.load(chkpt_path, deserialize=False)
    params = train_state.params
    if state is not None:
        params = jax.tree.map(lambda x: x[state], params)

    return params


def phys_conf_from_checkpoint(chkpt_path: Path) -> PhysicalConfiguration:
    r"""Load a :class:`~deepqmc.types.PhysicalConfiguration` from a checkpoint file.

    Reconstructs the electron and nuclear sample positions stored in the sampler
    state of a checkpoint, broadcasting the nuclear positions to match the
    electronic-state and electron-batch dimensions of the electron positions. Assumes
    a single molecule, i.e. the returned ``mol_idx`` is all zeros.

    Args:
        chkpt_path (~pathlib.Path): path to a ``chkpt-*.pt`` file written by
            :class:`~deepqmc.log.CheckpointStore`.

    Returns:
        ~deepqmc.types.PhysicalConfiguration: the electron and nuclear positions
        stored in the checkpoint's sampler state.
    """
    _step, train_state = CheckpointStore.load(chkpt_path, deserialize=False)

    r = train_state.sampler['elec']['r']
    R = train_state.sampler['nuc']['R'][
        :, None, None
    ]  # Electronic state and electron batch dims
    R = jnp.broadcast_to(R, (*r.shape[:-2], *R.shape[-2:]))

    return PhysicalConfiguration(R, r, jnp.zeros(r.shape[:-2], dtype=int))  # type: ignore
