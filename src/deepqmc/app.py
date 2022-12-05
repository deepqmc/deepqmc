import inspect
import logging
import pickle
import sys
import warnings
from pathlib import Path

import hydra
import jax
import yaml
from hydra.utils import call, get_original_cwd, to_absolute_path
from omegaconf import OmegaConf
from tqdm.auto import tqdm

__all__ = ()
log = logging.getLogger(__name__)

warnings.filterwarnings(
    'ignore',
    'provider=hydra.searchpath in main, path=conf is not available.',
    UserWarning,
)


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

    ansatz = instantiate_ansatz(hamil, ansatz)
    sampler = chain(*sampler[:-1], sampler[-1](hamil))
    return train(hamil, ansatz, sampler=sampler, **kwargs)


def train_from_checkpoint(workdir, restdir, evaluate, device, chkpt='LAST', **kwargs):
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


def main(cfg):
    log.info('Entering application')
    jax.config.update('jax_platform_name', cfg.device)
    log.info(f'Running on {cfg.device.upper()}')
    cfg.task.workdir = str(Path.cwd())
    log.info(f'Will work in {cfg.task.workdir}')
    maybe_log_code_version()
    call(cfg.task, device=cfg.device, _convert_='all')


@hydra.main(config_path='conf', config_name='config', version_base=None)
def cli(cfg):
    try:
        main(cfg)
    except hydra.errors.InstantiationException as e:
        raise e.__cause__ from None
    except KeyboardInterrupt:
        log.warning('Interrupted!')


def _get_subkwargs(func, name=None, mapping=None):
    target = mapping.get((func, name), False) if mapping is not None else func
    if not target:
        return {}
    target, override = target if isinstance(target, tuple) else (target, [])
    if isinstance(target, dict):
        sub_kwargs = {
            k: collect_kwarg_defaults(v) if callable(v) else v
            for k, v in target.items()
        }
    else:
        sub_kwargs = collect_kwarg_defaults(target)
    for x in override:
        if isinstance(x, tuple):
            key, val = x
            sub_kwargs[key] = val
        else:
            del sub_kwargs[x]
    return sub_kwargs


def collect_kwarg_defaults(func):
    from .fit import fit_wf
    from .gnn import SchNet
    from .pretrain import pretrain
    from .train import OPT_KWARGS, CheckpointStore, train
    from .wf import PauliNet
    from .wf.baseline import Baseline
    from .wf.paulinet.omni import Backflow, Jastrow, OmniNet

    DEEPQMC_DEFAULTS = {
        (train, 'pretrain_kwargs'): pretrain,
        (train, 'opt_kwargs'): OPT_KWARGS,
        (train, 'fit_kwargs'): fit_wf,
        (train, 'chkpts_kwargs'): CheckpointStore,
        (pretrain, 'baseline_kwargs'): Baseline.from_mol,
        (PauliNet.__init__, 'omni_kwargs'): OmniNet,
        (OmniNet.__init__, 'gnn_kwargs'): SchNet,
        (OmniNet.__init__, 'jastrow_kwargs'): Jastrow,
        (OmniNet.__init__, 'backflow_kwargs'): Backflow,
    }
    kwargs = {}
    func = func.__init__ if inspect.isclass(func) else func
    for p in inspect.signature(func).parameters.values():
        if p.kind != inspect.Parameter.KEYWORD_ONLY:
            continue

        if '_kwargs' in p.name:
            if DEEPQMC_DEFAULTS.get((func, p.name), False):
                sub_kwargs = _get_subkwargs(func, p.name, DEEPQMC_DEFAULTS)
                kwargs[p.name] = sub_kwargs

        else:
            if p.default is None:
                kwargs[p.name] = None
            elif p.default == inspect._empty:
                kwargs[p.name] = '???'
            else:
                try:
                    kwargs[p.name] = p.default
                except ValueError:
                    raise
    return kwargs


def collect_deepqmc_kwarg_defaults(workdir, device, return_yaml=False):
    import deepqmc
    import deepqmc.wf

    listed = [
        'MolecularHamiltonian',
        'Molecule',
        'train',
        'MetropolisSampler',
        'DecorrSampler',
        'ResampledSampler',
        'PauliNet',
    ]
    members = dict(inspect.getmembers(deepqmc.wf)) | dict(inspect.getmembers(deepqmc))
    funcs = {func: members[func] for func in listed}
    kwargs = {
        name: collect_kwarg_defaults(func)
        for (name, func) in funcs.items()
        if collect_kwarg_defaults(func)
    }
    log.info(
        'DeepQMC'
        f' defaults:\n{yaml.dump(kwargs, default_flow_style=False, sort_keys=False)}'
    )
    if return_yaml:
        with open('defaults.yaml', 'w') as outfile:
            yaml.dump(kwargs, outfile, default_flow_style=False, sort_keys=False)
