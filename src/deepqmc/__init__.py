import logging
import os
import sys

from omegaconf import OmegaConf

log = logging.getLogger(__name__)

if not os.environ.get('NVIDIA_TF32_OVERRIDE'):
    # disable TensorFloat-32 for better precision
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    if 'jax' in sys.modules:
        log.warning(
            'JAX was imported before deepqmc, TensorFloat32 precision might be enabled.'
            ' You may experience numerical issues.'
        )
elif os.environ.get('NVIDIA_TF32_OVERRIDE') != '0':
    log.warning(
        'TensorFloat-32 seems to be enabled. You might want to disable TensorFloat-32'
        ' precision by setting NVIDIA_TF32_OVERRIDE=0 before loading deepqmc to avoid'
        ' numerical issues.'
    )


from .hamil import MolecularHamiltonian  # noqa: E402
from .molecule import Molecule  # noqa: E402
from .sampling import (  # noqa: E402
    DecorrSampler,
    MetropolisSampler,
    ResampledSampler,
)
from .train import train  # noqa: E402

OmegaConf.register_new_resolver('eval', eval)

__all__ = [
    'DecorrSampler',
    'MetropolisSampler',
    'MolecularHamiltonian',
    'Molecule',
    'ResampledSampler',
    'train',
]
