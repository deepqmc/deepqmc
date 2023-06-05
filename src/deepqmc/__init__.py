from omegaconf import OmegaConf

from .hamil import MolecularHamiltonian
from .molecule import Molecule
from .sampling import DecorrSampler, MetropolisSampler, ResampledSampler
from .train import train


def eval_resolver(x):
    return eval(x)


OmegaConf.register_new_resolver('eval', eval_resolver)

__all__ = [
    'DecorrSampler',
    'MetropolisSampler',
    'MolecularHamiltonian',
    'Molecule',
    'ResampledSampler',
    'train',
]
