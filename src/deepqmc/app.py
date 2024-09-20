import logging
import platform
import sys
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Optional, Union

import hydra
from hydra.errors import InstantiationException
from hydra.utils import call, get_original_cwd, to_absolute_path
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from .molecule import Molecule, read_molecule_dataset
from .validate_kwargs import validate_kwargs

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


def read_molecules(
    directory: Union[Path, str, None] = None, whitelist: Optional[str] = None
) -> Optional[list[Molecule]]:
    if directory is None:
        return None
    path = Path(directory)
    if not path.is_absolute():
        path = to_absolute_path(get_original_cwd()) / path
    log.info(f'Reading molecules from {path}')
    molecules = read_molecule_dataset(path, whitelist)
    log.info(f'Read molecules from files: {", ".join(molecules.keys())}')
    log.info(f'Read {len(molecules)} molecules')
    if len(molecules) == 0:
        raise ValueError(
            f'No molecules found in {path}, with whitelist {whitelist!r}. '
            'Please check if task.mols.directory and task.mols.whitelist are correct.'
        )
    return list(molecules.values())


def instantiate_ansatz(hamil, ansatz):
    import haiku as hk

    return hk.without_apply_rng(
        hk.transform(
            lambda phys_conf, return_mos=False: ansatz(hamil)(phys_conf, return_mos)
        )
    )


def train_from_factories(hamil, ansatz, **kwargs):
    from .train import train

    ansatz = instantiate_ansatz(hamil, ansatz)
    return train(hamil, ansatz, **kwargs)


def assert_valid_restdir(restdir: Path, workdir: str):
    if not restdir.is_dir():
        raise ValueError(f'restdir {restdir!r} is not a directory')
    # restdir is workdir/{training/evaluation}
    if str(restdir.parent) == workdir:
        raise ValueError(
            'Cannot restore from the same directory as the one you are running in. '
            'Make sure that task.restdir and hydra.run.dir are different.'
        )


def train_from_checkpoint(workdir, restdir, evaluate, chkpt='LAST', **kwargs):
    restdir = Path(to_absolute_path(get_original_cwd())) / restdir
    assert_valid_restdir(restdir, workdir)
    cfg, step, train_state = task_from_workdir(restdir, chkpt)
    while cfg.task.get('restdir', False):
        restdir = Path(to_absolute_path(get_original_cwd())) / cfg.task.restdir
        assert_valid_restdir(restdir, workdir)
        cfg, *_ = task_from_workdir(restdir, 'LAST')
    log.info(f'Found original config file in {restdir}')
    cfg.task.workdir = workdir
    if not kwargs.pop('keep_sampler_state', not evaluate):
        train_state = train_state._replace(sampler=None)
    if evaluate:
        cfg.task.opt = None
        train_state = train_state._replace(opt=None)
    else:
        cfg.task.init_step = step
    cfg = OmegaConf.to_object(cfg)
    assert isinstance(cfg, dict)
    call(cfg['task'], _convert_='all', train_state=train_state, **kwargs)


def task_from_workdir(workdir, chkpt):
    from .train import CheckpointStore

    workdir = Path(workdir)
    assert workdir.is_dir()
    cfg = OmegaConf.load(workdir / '.hydra' / 'config.yaml')
    if chkpt == 'LAST':
        chkpts = list(workdir.glob(CheckpointStore.PATTERN.format('*')))
        if not chkpts:
            chkpts = (workdir / 'training').glob(CheckpointStore.PATTERN.format('*'))
        chkpt = sorted(
            chkpts,
            key=lambda path: CheckpointStore.extract_step_from_filename(path.name),
        )[-1]
    else:
        chkpt = workdir / chkpt
    step, train_state = CheckpointStore.load(chkpt)
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

        cwd = Path(__file__).resolve().parent

        def git_command(command):
            return subprocess.check_output(['git'] + command, cwd=cwd).strip().decode()

        try:
            sha = git_command(['rev-parse', '--short', 'HEAD'])
            diff = git_command(['diff'])
        except Exception:
            sha = 'deepqmc ' + version('deepqmc')
            diff = None
        log.debug(f'Running with code version: {sha}')
        if diff:
            log.debug(f'With uncommitted changes:\n{diff}')


def detect_devices():
    import jax

    device_kinds = [device.device_kind for device in jax.devices()]
    assert all(dk == device_kinds[0] for dk in device_kinds)
    n_device = len(device_kinds)
    n_process = jax.process_count()
    log.info(f'Process {jax.process_index()} running on {platform.node()}')
    log.info(
        'Running on'
        f' {n_device} {device_kinds[0].upper()}{"" if n_device == 1 else "s"} with'
        f' {n_process} process{"" if n_process == 1 else "es"}'
    )


def main(cfg):
    assert log.parent is not None
    log.parent.setLevel(cfg.logging.deepqmc)
    logging.getLogger('jax').setLevel(cfg.logging.jax)
    logging.getLogger('absl').setLevel(cfg.logging.kfac)
    log.info('Entering application')
    detect_devices()
    cfg.task.workdir = str(Path.cwd())
    log.info(f'Will work in {cfg.task.workdir}')
    maybe_log_code_version()
    cfg = OmegaConf.to_object(cfg)
    assert isinstance(cfg, dict)
    validate_kwargs(cfg['task'])
    call(cfg['task'], _convert_='all')


@hydra.main(config_path='conf', config_name='config', version_base=None)
def cli(cfg):
    try:
        main(cfg)
    except InstantiationException as e:
        raise e.__cause__ from None  # type: ignore
    except KeyboardInterrupt:
        log.warning('Interrupted!')
