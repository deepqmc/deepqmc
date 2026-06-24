from pathlib import Path

import jax
import jax.numpy as jnp

from ..log import CheckpointStore
from ..types import Params, PhysicalConfiguration


def load_parameters(
    chkpt_path: Path, squeeze_electronic_states: bool = False
) -> Params:
    r"""Load ansatz parameters from a checkpoint file.

    Args:
        chkpt_path (~pathlib.Path): path to a ``chkpt-*.pt`` file written by
            :class:`~deepqmc.log.CheckpointStore`.
        squeeze_electronic_states (bool): optional, if :data:`True` removes the
            leading electronic-state axis of the loaded parameters. Useful when
            the parameters should be used with an ansatz instantiated without an
            explicit state axis, e.g. via
            :func:`~deepqmc.postprocess.ansatz_utils.instantiate_predefined_ansatz`.

    Returns:
        ~deepqmc.types.Params: the ansatz parameters stored in the checkpoint.
    """
    _step, train_state = CheckpointStore.load(chkpt_path, deserialize=False)
    params = train_state.params
    if squeeze_electronic_states:
        params = jax.tree.map(lambda x: x.squeeze(axis=0), params)

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
