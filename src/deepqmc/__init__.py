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

del log

import jax  # noqa: E402

from .parallel import maybe_init_multi_host  # noqa: E402

maybe_init_multi_host()

from .conf.custom_resolvers import mode_subdir, process_idx_suffix  # noqa: E402

jax.config.update('jax_default_matmul_precision', 'highest')
OmegaConf.register_new_resolver('eval', eval)
OmegaConf.register_new_resolver('process_idx_suffix', process_idx_suffix)
OmegaConf.register_new_resolver('mode_subdir', mode_subdir)
