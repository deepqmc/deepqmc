import logging
import pickle
import sys
import warnings
from glob import glob
from pathlib import Path

import hydra
import yaml
from hydra.utils import call, get_original_cwd, to_absolute_path
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from deepqmc import Molecule

__all__ = ()
log = logging.getLogger(__name__)

warnings.filterwarnings(
    'ignore',
    'provider=hydra.searchpath in main, path=conf is not available.',
    UserWarning,
)
warnings.filterwarnings(
    'ignore',
    'Some donated buffers were not usable:',
    UserWarning,
)
warnings.filterwarnings(
    'ignore',
    'Explicitly requested dtype',
    UserWarning,
)


def read_molecules(directory):
    if directory is None:
        return None
    path = Path(directory)
    if not path.is_absolute():
        path = to_absolute_path(get_original_cwd()) / path
    log.info(f'Reading molecules from {path}')
    molecules = []
    for f in sorted(glob(str(path / '*.yaml'))):
        with open(f, 'r') as stream:
            molecules.append(Molecule(**yaml.safe_load(stream)))
    log.info(f'Read {len(molecules)} molecules')
    return molecules


def instantiate_ansatz(hamil, ansatz):
    import haiku as hk

    envelope = ansatz.keywords['envelope']
    if envelope.keywords.get('is_baseline', False):
        log.warning(
            '\x1b[31;20m DeprecationWarning - Running the original PauliNet is'
            ' deprecated and should be used only for the purpose of reproducing old'
            ' results. Please consider using one of the other architectures to obtain'
            ' accurate results. For more details on the performance of different models'
            ' see the implementation paper.\x1b[0m'
        )
        # envelope returns a partial(Baseline, **baseline_params) object
        ansatz.keywords['envelope'] = envelope(hamil.mol, hamil)

    return hk.without_apply_rng(
        hk.transform(
            lambda phys_conf, return_mos=False: ansatz(hamil)(phys_conf, return_mos)
        )
    )


def train_from_factories(hamil, ansatz, sampler, **kwargs):
    from .sampling import chain
    from .train import train

    ansatz = instantiate_ansatz(hamil, ansatz)
    sampler = chain(*sampler[:-1], sampler[-1](hamil))
    return train(hamil, ansatz, sampler=sampler, **kwargs)


def train_from_checkpoint(workdir, restdir, evaluate, chkpt='LAST', **kwargs):
    restdir = Path(to_absolute_path(get_original_cwd())) / restdir
    if not restdir.is_dir():
        raise ValueError(f'restdir {restdir!r} is not a directory')
    cfg, step, train_state = task_from_workdir(restdir, chkpt)
    while cfg.task.get('restdir', False):
        restdir = Path(to_absolute_path(get_original_cwd())) / cfg.task.restdir
        cfg, *_ = task_from_workdir(restdir, chkpt)
    log.info(f'Found original config file in {restdir}')
    cfg.task.workdir = workdir
    if evaluate:
        cfg.task.opt = None
    else:
        cfg.task.init_step = step
    cfg = OmegaConf.to_object(cfg)
    call(cfg['task'], _convert_='all', train_state=train_state, **kwargs)


def task_from_workdir(workdir, chkpt):
    from .train import CheckpointStore

    workdir = Path(workdir)
    assert workdir.is_dir()
    cfg = OmegaConf.load(workdir / '.hydra/config.yaml')
    if chkpt == 'LAST':
        chkpts = list(workdir.glob(CheckpointStore.PATTERN.format('*')))
        if not chkpts:
            chkpts = (workdir / 'training').glob(CheckpointStore.PATTERN.format('*'))
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


def maybe_log_code_version():
    if log.isEnabledFor(logging.DEBUG):
        import subprocess

        def git_command(command):
            return (
                subprocess.check_output(
                    ['git'] + command, cwd=Path(__file__).resolve().parent
                )
                .strip()
                .decode()
            )

        sha = git_command(['rev-parse', '--short', 'HEAD'])
        diff = git_command(['diff'])
        log.debug(f'Running with code version: {sha}')
        if diff:
            log.debug(f'With uncommitted changes:\n{diff}')


def detect_devices():
    import jax

    device_kinds = [device.device_kind for device in jax.devices()]
    assert all(dk == device_kinds[0] for dk in device_kinds)
    n_device = len(device_kinds)
    n_process = jax.process_count()
    log.info(
        'Running on'
        f' {n_device} {device_kinds[0].upper()}{"" if n_device == 1 else "s"} with'
        f' {n_process} process{"" if n_process == 1 else "es"}'
    )


def main(cfg):
    log.setLevel(cfg.logging.deepqmc)
    logging.getLogger('jax').setLevel(cfg.logging.jax)
    logging.getLogger('absl').setLevel(cfg.logging.kfac)
    log.info('Entering application')
    detect_devices()
    cfg.task.workdir = str(Path.cwd())
    log.info(f'Will work in {cfg.task.workdir}')
    maybe_log_code_version()
    cfg = OmegaConf.to_object(cfg)
    call(cfg['task'], _convert_='all')


@hydra.main(config_path='conf', config_name='config', version_base=None)
def cli(cfg):
    try:
        main(cfg)
    except hydra.errors.InstantiationException as e:
        raise e.__cause__ from None
    except KeyboardInterrupt:
        log.warning('Interrupted!')
