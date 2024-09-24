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
    """
    with resources.as_file(resources.files('deepqmc.conf')) as conf_dir:
        with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
            cfg = compose(config_name='config', overrides=[f'ansatz={name}'])

    _ansatz = instantiate(cfg['ansatz'], _recursive_=True, _convert_='all')

    return instantiate_ansatz(H, _ansatz)


def instantiate_wf_from_checkpoint(
    ansatz_name: str, H: MolecularHamiltonian, chkpt_path: Path
) -> WaveFunction:
    r"""Instantiate a predefined ansatz and loads its parameter from a checkpoint file.

    See also :func:`~deepqmc.postprocess.ansatz_utils.instantiate_predefined_ansatz`
    and :func:`~deepqmc.postprocess.checkpoint_utils.load_parameters`.
    """
    ansatz = instantiate_predefined_ansatz(ansatz_name, H)
    params = load_parameters(chkpt_path, squeeze_electronic_states=True)
    return partial(ansatz.apply, params)
