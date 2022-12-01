import logging
import pickle
import sys
from pathlib import Path

import jax
from hydra.utils import call, get_original_cwd, to_absolute_path
from omegaconf import OmegaConf
from tqdm.auto import tqdm

__all__ = ()
log = logging.getLogger(__name__)


def move_to(device):
    jax.config.update('jax_platform_name', device)
    log.info(f'Running on {device.upper()}')


def instantiate_ansatz(hamil, ansatz):
    import haiku as hk

    return hk.without_apply_rng(
        hk.transform_with_state(
            lambda r, return_mos=False: ansatz(hamil)(r, return_mos)
        )
    )


def train_from_factories(hamil, ansatz, sampler, device, **kwargs):
    from .sampling import chain
    from .train import train

    move_to(device)
    ansatz = instantiate_ansatz(hamil, ansatz)
    sampler = chain(*sampler[:-1], sampler[-1](hamil))
    return train(hamil, ansatz, sampler=sampler, **kwargs)


def train_from_checkpoint(workdir, restdir, evaluate, device, chkpt='LAST', **kwargs):
    move_to(device)
    restdir = Path(to_absolute_path(get_original_cwd())) / restdir
    if not restdir.is_dir():
        raise ValueError(f'restdir "{restdir}" is not a directory')
    cfg, step, train_state = task_from_workdir(restdir, chkpt)
    cfg.task.workdir = workdir
    if evaluate:
        cfg.task.opt = None
    else:
        cfg.task.init_step = step
    call(cfg.task, _convert_='all', train_state=train_state, **kwargs)


def task_from_workdir(workdir, chkpt, device=None):
    from .train import CheckpointStore

    workdir = Path(workdir)
    assert workdir.is_dir()
    cfg = OmegaConf.load(workdir / '.hydra/config.yaml')
    if chkpt == 'LAST':
        chkpts = list(workdir.glob(CheckpointStore.PATTERN.format('*')))
        if not chkpts:
            chkpts = (workdir / 'train').glob(CheckpointStore.PATTERN.format('*'))
        chkpt = sorted(chkpts)[-1]
    else:
        chkpt = workdir / chkpt
    with open(chkpt, 'rb') as f:
        step, train_state = pickle.load(f)
    return cfg, step, train_state


class TqdmStream:
    @staticmethod
    def write(msg: str) -> int:
        try:
            tqdm.write(msg, end='')
        except BrokenPipeError:
            sys.stderr.write(msg)
            return 0
        return len(msg)
