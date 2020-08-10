import importlib
import logging
import shutil
from functools import partial

import toml
import torch

from .errors import TomlError
from .molecule import Molecule
from .wf import PauliNet
from .wf.paulinet.omni import OmniSchNet

log = logging.getLogger(__name__)

__all__ = ()


def validate_params(params):
    REQUIRED = {'system'}
    OPTIONAL = {'model_kwargs', 'train_kwargs', 'evaluate_kwargs'}
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
    pyscf_file = workdir / 'baseline.pyscf'
    system = params.pop('system')
    if isinstance(system, str):
        name, system = system, {}
    else:
        name = system.pop('name')
    if ':' in name:
        mol = import_fullname(name)(**system)
    else:
        mol = Molecule.from_name(name, **system)
    model_kwargs = params.pop('model_kwargs', {})
    if pyscf_file.is_file():
        mf, mc = pyscf_from_file(pyscf_file)
        log.info(f'Restored PySCF object from {pyscf_file}')
        # TODO refactor initialisation to avoid duplicate with PauliNet.from_hf
        assert mf.mol.basis == model_kwargs.pop('basis', '6-311g')
        cas = model_kwargs.pop('cas', None)
        assert not mc and not cas or (mc.ncas == cas[0] and sum(mc.nelecas) == cas[1])
        omni_kwargs = model_kwargs.pop('omni_kwargs', None)
        pauli_kwargs = model_kwargs.pop('pauli_kwargs', None)
        assert not model_kwargs
        wf = PauliNet.from_pyscf(
            mc or mf,
            **{
                'omni_factory': partial(OmniSchNet, **(omni_kwargs or {})),
                'cusp_correction': True,
                'cusp_electrons': True,
                **(pauli_kwargs or {}),
            },
        )
        wf.mf = mf
    else:
        wf = PauliNet.from_hf(mol, **model_kwargs)
        shutil.copy(wf.mf.chkfile, pyscf_file)
    return wf, params, state


def pyscf_from_file(chkfile):
    import pyscf.gto.mole
    from pyscf import lib, mcscf, scf

    pyscf.gto.mole.float32 = float

    mol = lib.chkfile.load_mol(chkfile)
    mf = scf.RHF(mol)
    mf.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
    mc_dict = lib.chkfile.load(chkfile, 'mcscf')
    if mc_dict:
        mc_dict['ci'] = lib.chkfile.load(chkfile, 'ci')
        mc_dict['nelecas'] = tuple(map(int, lib.chkfile.load(chkfile, 'nelecas')))
        mc = mcscf.CASSCF(mf, 0, 0)
        mc.__dict__.update(mc_dict)
    else:
        mc = None
    return mf, mc
