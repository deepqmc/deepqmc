import logging
import sys
from pathlib import Path

import hydra
import hydra.errors
import torch
from hydra import compose, initialize
from hydra.utils import call, instantiate, to_absolute_path
from omegaconf import OmegaConf
from tqdm.auto import tqdm

__all__ = ()

log = logging.getLogger(__name__)


def eval_resolver(x):
    import numpy as np  # noqa: F401
    import torch  # noqa: F401

    return eval(x)


OmegaConf.register_new_resolver('cls', hydra.utils.get_class)
OmegaConf.register_new_resolver('fn', hydra.utils.get_method)
OmegaConf.register_new_resolver('eval', eval_resolver)


def ansatz_from_name(name, mol, **kwargs):
    with initialize('conf/ansatz'):
        ansatz = compose(config_name=name)
    for k, v in kwargs.items():
        OmegaConf.update(ansatz, k, v)
    ansatz.mol = None
    ansatz = OmegaConf.to_object(ansatz)
    ansatz['mol'] = mol
    return instantiate(ansatz)


def ansatz_from_workdir(workdir, state='LAST'):
    workdir = Path(workdir)
    cfg, state = from_workdir(workdir, state)
    cfg.workdir = None
    del cfg.task.state
    cfg = instantiate_wf(cfg, workdir, state)
    return cfg['task']['wf']


def from_workdir(workdir, state):
    cfg = OmegaConf.load(workdir / '.hydra/config.yaml')
    if 'workdir' in cfg.ansatz:
        cfg.ansatz.workdir = str(workdir)
    if state == 'LAST':
        state = sorted(workdir.glob('**/state-*.pt'))[-1]
    else:
        state = workdir / state
    return cfg, state


def instantiate_wf(cfg, fromdir, state):
    cfg = OmegaConf.to_object(cfg)
    cfg['task']['wf'] = instantiate(cfg['task']['wf'])
    if fromdir:
        log.info(f'Loading state from {state}...')
        state_dict = torch.load(state)
        log.info('State loaded')
        if 'state' in cfg['task']:
            assert cfg['task']['state'] is None
            cfg['task']['state'] = state_dict
        else:
            cfg['task']['wf'].load_state_dict(state_dict['wf'])
    return cfg


@hydra.main(config_path='conf', config_name='config')
def cli(cfg):
    log.info('Entering application')
    cfg.workdir = str(Path.cwd())
    fromdir = OmegaConf.select(cfg, 'fromdir', throw_on_missing=True)
    state = None
    if fromdir:
        fromdir = Path(to_absolute_path(fromdir))
        cfg_orig, state = from_workdir(fromdir, cfg.state)
        cfg_orig = OmegaConf.masked_copy(cfg_orig, cfg_orig.keys() - cfg.keys())
        cfg = OmegaConf.merge(cfg_orig, cfg)
    try:
        if cfg.seed is not None:
            log.info(f'Setting random seed: {cfg.seed}')
            torch.manual_seed(cfg.seed)
        device = cfg.device
        if cfg.anomaly:
            torch.autograd.set_detect_anomaly(True)
            log.warn('Setting anomaly detection on')
        cfg = instantiate_wf(cfg, fromdir, state)
        log.info(f'Moving to {device}...')
        cfg['task']['wf'].to(device)
        log.info(f'Moved to {device}')
        call(cfg['task'], _convert_='all')
    except hydra.errors.InstantiationException as e:
        raise e.__cause__ from None
    except KeyboardInterrupt:
        log.warning('Interrupted!')


class TqdmStream:
    def write(self, msg: str) -> int:
        try:
            tqdm.write(msg, end='')
        except BrokenPipeError:
            sys.stderr.write(msg)
            return 0
        return len(msg)


tqdmout = TqdmStream()
