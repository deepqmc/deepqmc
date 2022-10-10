import logging
import pickle
import sys
from pathlib import Path

import hydra
import hydra.errors
from hydra.utils import call, instantiate
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from .sampling import chain

__all__ = ()
log = logging.getLogger(__name__)


def instantiate_ansatz(hamil, ansatz):
    import haiku as hk

    return hk.without_apply_rng(hk.transform_with_state(lambda r: ansatz(hamil)(r)))


def train_from_factories(hamil, ansatz, sampler, **kwargs):
    from .train import train

    ansatz = instantiate_ansatz(hamil, ansatz)
    sampler = chain(*sampler[:-1], sampler[-1](hamil))
    return train(hamil, ansatz, sampler=sampler, **kwargs)


def task_from_workdir(workdir, chkpt='LAST', device=None):
    from .train import CheckpointStore

    workdir = Path(workdir)
    assert workdir.is_dir()
    cfg = OmegaConf.load(workdir / '.hydra/config.yaml')
    hamil = instantiate(cfg.task.hamil)
    ansatz = instantiate_ansatz(hamil, cfg.task.ansatz)
    chkpt = (
        sorted(workdir.glob(CheckpointStore.PATTERN.format('*')))[-1]
        if chkpt == 'LAST'
        else workdir / chkpt
    )
    with open(chkpt, 'rb') as f:
        train_state = pickle.load(f)
    return hamil, ansatz, train_state


def main(cfg):
    log.info('Entering application')
    cfg.task.workdir = str(Path.cwd())
    log.info(f'Will work in {cfg.task.workdir}')
    call(cfg.task, _convert_='all')


@hydra.main(config_path='conf', config_name='config', version_base=None)
def cli(cfg):
    try:
        main(cfg)
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


if __name__ == '__main__':
    cli()
