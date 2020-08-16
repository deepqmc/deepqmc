import importlib
import logging

import toml
import torch

from .errors import TomlError
from .molecule import Molecule
from .wf import ANSATZES

log = logging.getLogger(__name__)

__all__ = ()


def validate_params(params):
    REQUIRED = {'system', 'ansatz'}
    OPTIONAL = {'train_kwargs', 'evaluate_kwargs'} | {f'{a}_kwargs' for a in ANSATZES}
    params = set(params)
    missing = REQUIRED - params
    if missing:
        raise TomlError(f'Missing keywords: {missing}')
    unknown = params - REQUIRED - OPTIONAL
    if unknown:
        raise TomlError(f'Unknown keywords: {unknown}')


def import_fullname(fullname):
    module_name, qualname = fullname.split(':')
    module = importlib.import_module(module_name)
    return getattr(module, qualname)


def wf_from_file(workdir):
    params = toml.loads((workdir / 'param.toml').read_text())
    validate_params(params)
    state_file = workdir / 'state.pt'
    state = torch.load(state_file) if state_file.is_file() else None
    if state:
        log.info(f'State loaded from {state_file}')
    system = params.pop('system')
    if isinstance(system, str):
        name, system = system, {}
    else:
        name = system.pop('name', None)
    if name is None:
        mol = Molecule(**system)
    elif ':' in name:
        mol = import_fullname(name)(**system)
    else:
        mol = Molecule.from_name(name, **system)
    ansatz = params.pop('ansatz')
    ansatz = ANSATZES[ansatz]
    kwargs = params.pop(f'{ansatz.name}_kwargs', {})
    if ansatz.uses_workdir:
        assert 'workdir' not in kwargs
        kwargs['workdir'] = workdir
    wf = ansatz.entry(mol, **kwargs)
    return wf, params, state
