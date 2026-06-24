from functools import partial
from importlib import resources
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from ..app import instantiate_ansatz
from ..hamil import MolecularHamiltonian
from ..types import Ansatz, WaveFunction
from .checkpoint_utils import load_parameters


def instantiate_predefined_ansatz(name: str, H: MolecularHamiltonian) -> Ansatz:
    r"""Instantiate one of the predefined ansatzes.

    The hydra configuration file ``name.yaml`` must be present in the
    ``src/deepqmc/conf/ansatz`` directory.

    Args:
        name (str): the name of the predefined ansatz configuration, e.g.
            ``'psiformer'`` or ``'transpsiformer'``.
        H (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the system the
            ansatz is instantiated for.

    Returns:
        ~deepqmc.types.Ansatz: the instantiated wave function ansatz, with
            uninitialized parameters.
    """
    with resources.as_file(resources.files('deepqmc.conf')) as conf_dir:
        with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
            cfg = compose(config_name='config', overrides=[f'ansatz={name}'])

    _ansatz = instantiate(cfg['ansatz'], _recursive_=True, _convert_='all')

    return instantiate_ansatz(H, _ansatz)


def instantiate_wf_from_checkpoint(
    ansatz_name: str, H: MolecularHamiltonian, chkpt_path: Path
) -> WaveFunction:
    r"""Instantiate a predefined ansatz and load its parameters from a checkpoint file.

    Convenience function combining
    :func:`~deepqmc.postprocess.ansatz_utils.instantiate_predefined_ansatz` and
    :func:`~deepqmc.postprocess.checkpoint_utils.load_parameters`, to obtain a
    ready-to-evaluate :data:`~deepqmc.types.WaveFunction` from a finished training
    run.

    Args:
        ansatz_name (str): the name of the predefined ansatz configuration that was
            used for training, e.g. ``'psiformer'`` or ``'transpsiformer'``.
        H (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the system the
            ansatz was trained for.
        chkpt_path (~pathlib.Path): path to a ``chkpt-*.pt`` file written by
            :class:`~deepqmc.log.CheckpointStore`.

    Returns:
        ~deepqmc.types.WaveFunction: the trained wave function, with its parameters
            already bound, ready to be evaluated on a
            :class:`~deepqmc.types.PhysicalConfiguration`.
    """
    ansatz = instantiate_predefined_ansatz(ansatz_name, H)
    params = load_parameters(chkpt_path, squeeze_electronic_states=True)
    return partial(ansatz.apply, params)
