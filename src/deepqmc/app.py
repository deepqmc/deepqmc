import logging
import platform
import sys
import warnings
from pathlib import Path
from typing import Optional

import hydra
from hydra.errors import InstantiationException
from hydra.utils import call, get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from .hamil import MolecularHamiltonian
from .molecule import Molecule, read_molecule_dataset
from .types import Ansatz, AnsatzFactory, TrainState
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
    directory: Path | str | None = None, whitelist: Optional[str] = None
) -> Optional[list[Molecule]]:
    r"""Read a dataset of molecules for transferable training.

    Reads every molecule ``.yaml`` file below :data:`directory` (see
    :func:`~deepqmc.molecule.read_molecule_dataset`), optionally restricted to
    filenames matching :data:`whitelist`. Meant to be used as the ``mols`` entry
    of a training config, e.g. via ``_target_: deepqmc.app.read_molecules``.

    Args:
        directory (Optional[str | ~pathlib.Path]): the directory containing the
            molecule ``.yaml`` files; relative paths are resolved against the
            original working directory. If :data:`None`, no molecules are read.
        whitelist (Optional[str]): optional regular expression; only molecule
            files whose name matches it are read.
        verbose (bool): optional, whether to log the molecules that were found.

    Returns:
        Optional[list[~deepqmc.molecule.Molecule]]: the molecules found in
        :data:`directory`, or :data:`None` if :data:`directory` is :data:`None`.
    """
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


def instantiate_ansatz(hamil: MolecularHamiltonian, ansatz: AnsatzFactory) -> Ansatz:
    r"""Instantiate a wave function :class:`~deepqmc.types.Ansatz` for a Hamiltonian.

    Wraps the given :data:`~deepqmc.types.AnsatzFactory` in a
    :func:`haiku.transform`, producing an object with ``init`` and ``apply``
    methods that can be used to initialize and evaluate the wave function
    (see the :ref:`tutorial <tutorial>`).

    Args:
        hamil (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the
            physical system the ansatz is instantiated for.
        ansatz (~deepqmc.types.AnsatzFactory): a callable that returns an
            uninstantiated wave function model when called with :data:`hamil`.

    Returns:
        ~deepqmc.types.Ansatz: the instantiated wave function ansatz.
    """
    import haiku as hk

    return hk.without_apply_rng(
        hk.transform(
            lambda phys_conf, return_mos=False: ansatz(hamil)(phys_conf, return_mos)  # type: ignore
        )
    )


def train_from_factories(
    hamil: MolecularHamiltonian, ansatz: AnsatzFactory, **kwargs
) -> TrainState:
    r"""Instantiate the Ansatz and start training or evaluation.

    Convenience wrapper combining :func:`instantiate_ansatz` and
    :func:`~deepqmc.train.train`. This is the function invoked by the default
    ``train`` hydra task configs, via ``_target_: deepqmc.app.train_from_factories``.

    Args:
        hamil (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the
            physical system.
        ansatz (~deepqmc.types.AnsatzFactory): a callable that returns an
            uninstantiated wave function model when called with :data:`hamil`.
        kwargs: further keyword arguments forwarded to
            :func:`~deepqmc.train.train`.

    Returns:
        ~deepqmc.types.TrainState: the final training/evaluation state, as
        returned by :func:`~deepqmc.train.train`.
    """
    from .train import train

    instantiated_ansatz = instantiate_ansatz(hamil, ansatz)
    return train(hamil, instantiated_ansatz, **kwargs)


def assert_valid_restdir(restdir: Path, workdir: str):
    if not restdir.is_dir():
        raise ValueError(f'restdir {restdir!r} is not a directory')
    # restdir is workdir/{training/evaluation}
    if str(restdir.parent) == workdir:
        raise ValueError(
            'Cannot restore from the same directory as the one you are running in. '
            'Make sure that task.restdir and hydra.run.dir are different.'
        )


def train_from_checkpoint(
    workdir: str, restdir: str, evaluate: bool, chkpt='LAST', **kwargs
):
    r"""Restore a previous run and continue training or run evaluation.

    Restores the hydra task config and the :class:`~deepqmc.types.TrainState`
    checkpoint from :data:`restdir`, following the chain of ``restdir``
    references if the run in :data:`restdir` was itself restored from an
    earlier one, and re-invokes the restored task with the restored state. This
    is the function invoked by the ``restart`` and ``evaluate`` hydra task
    configs, via ``_target_: deepqmc.app.train_from_checkpoint``.

    Args:
        workdir (str): the working directory of the current job (supplied by
            hydra); used to derive the training/evaluation subdirectory and to
            guard against restoring from the directory currently being written to.
        restdir (str): the working directory of a previous run to restore from;
            if relative, it is resolved against the original working directory.
        evaluate (bool): if :data:`True`, run evaluation only: the optimizer
            state is dropped and the restored ansatz is no longer updated.
        chkpt (str): optional, the name of the checkpoint file to restore, or
            ``'LAST'`` (default) to restore the most recent checkpoint found in
            :data:`restdir`.
        kwargs: keyword arguments overriding those of the restored task config,
            e.g. ``keep_sampler_state`` to control whether the sampler state is
            also restored.
    """
    restdir_path = Path(restdir)
    if not restdir_path.is_absolute():
        restdir_path = Path(to_absolute_path(get_original_cwd())) / restdir
    cfg, step, train_state, task_overrides = task_from_chain_of_workdirs(
        workdir, restdir_path, chkpt
    )
    cfg.task.workdir = workdir
    kwargs = {**OmegaConf.to_object(task_overrides), **kwargs}  # type: ignore
    if not kwargs.pop('keep_sampler_state', not evaluate):
        train_state = train_state._replace(sampler=None)
    if evaluate:
        cfg.task.opt = None
        train_state = train_state._replace(opt=None)
    else:
        cfg.task.init_step = step
    cfg = OmegaConf.to_object(cfg)
    assert isinstance(cfg, dict)
    call(cfg['task'], _convert_='all', train_state=train_state, **kwargs)  # type: ignore


def task_from_chain_of_workdirs(workdir: str, restdir: Path, chkpt: str):
    assert_valid_restdir(restdir, workdir)
    cfg, step, train_state, next_restdir, task_overrides = task_from_workdir(
        restdir, chkpt, DictConfig({})
    )
    assert train_state is not None
    while next_restdir:
        restdir = (
            next_restdir
            if next_restdir.is_absolute()
            else Path(to_absolute_path(get_original_cwd())) / next_restdir
        )
        assert_valid_restdir(restdir, workdir)
        cfg, _, _, next_restdir, task_overrides = task_from_workdir(
            restdir, 'LAST', task_overrides
        )
    log.info(f'Found original config file in {restdir}, from checkpoint {chkpt}')
    return cfg, step, train_state, task_overrides


def update_task_overrides(
    cfg: DictConfig,
    task_overrides: DictConfig,
) -> DictConfig:
    updated_task_overrides = OmegaConf.merge(
        DictConfig(
            {
                key: cfg.task[key]
                for key in cfg.task.keys()
                if key not in ['keep_sampler_state', 'workdir', '_target_', 'chkpt']
            }
        ),
        task_overrides,
    )
    assert isinstance(updated_task_overrides, DictConfig)
    return updated_task_overrides


def task_from_workdir(
    workdir: Path, chkpt: str, task_overrides: DictConfig
) -> tuple[DictConfig, int, Optional[TrainState], Optional[Path], DictConfig]:
    from .train import CheckpointStore

    workdir = Path(workdir)
    assert workdir.is_dir()
    cfg = OmegaConf.load(workdir / '.hydra' / 'config.yaml')
    assert isinstance(cfg, DictConfig), 'DeepQMC config should always be a DictConfig.'
    assert not cfg.task.pop(
        'evaluate', False
    ), f'Cannot restart from evaluation job in {workdir}.'
    restdir = cfg.task.pop('restdir', None)
    if restdir:
        task_overrides = update_task_overrides(cfg, task_overrides)
        restdir = Path(restdir)
    try:
        if chkpt == 'LAST':
            chkpts = workdir.glob(CheckpointStore.PATTERN.format('*'))
            if not chkpts:
                chkpts = (workdir / 'training').glob(
                    CheckpointStore.PATTERN.format('*')
                )
            chkpt_path = sorted(
                chkpts,
                key=lambda path: CheckpointStore.extract_step_from_filename(path.name),
            )[-1]
        else:
            chkpt_path = workdir / chkpt
        step, train_state = CheckpointStore.load(chkpt_path)
    except Exception:
        # No checkpoint found, continue without train state
        step = 0
        train_state = None
    return cfg, step, train_state, restdir, task_overrides


class TqdmStream:
    @staticmethod
    def write(msg: str) -> int:
        try:
            tqdm.write(msg, end='')
        except BrokenPipeError:
            sys.stderr.write(msg)
            return 0
        return len(msg)


def maybe_log_code_version() -> None:
    if log.isEnabledFor(logging.DEBUG):
        import subprocess

        cwd = Path(__file__).resolve().parent

        def git_command(command):
            return subprocess.check_output(['git'] + command, cwd=cwd).strip().decode()

        try:
            sha = git_command(['rev-parse', '--short', 'HEAD'])
            diff = git_command(['diff'])
        except Exception:
            sha = 'deepqmc 1.2.0'
            diff = None
        log.debug(f'Running with code version: {sha}')
        if diff:
            log.debug(f'With uncommitted changes:\n{diff}')


def detect_devices() -> None:
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
    if not cfg['task'].get('restdir', False):
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
