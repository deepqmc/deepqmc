import importlib
import logging
import sys
import time
from pathlib import Path

import click
import toml
import tomlkit
import torch
from torch.utils.tensorboard import SummaryWriter

from .defaults import DEEPQMC_MAPPING, collect_kwarg_defaults
from .errors import TrainingCrash
from .evaluate import evaluate
from .molecule import Molecule
from .train import train
from .wf import PauliNet

log = logging.getLogger(__name__)


def import_fullname(fullname):
    module_name, qualname = fullname.split(':')
    module = importlib.import_module(module_name)
    return getattr(module, qualname)


def wf_from_file(path):
    params = toml.loads(Path(path).read_text())
    system = params.pop('system')
    if isinstance(system, str):
        name, system = system, {}
    else:
        name = system.pop('name')
    if ':' in name:
        mol = import_fullname(name)(**system)
    else:
        mol = Molecule.from_name(name, **system)
    wf = PauliNet.from_hf(mol, **params.pop('model_kwargs', {}))
    return wf, params


@click.group()
def cli():
    logging.basicConfig(style='{', format='{message}', datefmt='%H:%M:%S')
    logging.getLogger('deepqmc').setLevel(logging.DEBUG)


@cli.command()
@click.option('--commented', '-c', is_flag=True)
def defaults(commented):
    table = tomlkit.table()
    table['model_kwargs'] = collect_kwarg_defaults(PauliNet.from_hf, DEEPQMC_MAPPING)
    table['train_kwargs'] = collect_kwarg_defaults(train, DEEPQMC_MAPPING)
    table['evaluate_kwargs'] = collect_kwarg_defaults(evaluate, DEEPQMC_MAPPING)
    lines = tomlkit.dumps(table).split('\n')
    if commented:
        lines = ['# ' + l if ' = ' in l and l[0] != '#' else l for l in lines]
    click.echo('\n'.join(lines), nl=False)


@cli.command('train')
@click.argument('workdir', type=click.Path(exists=True))
@click.option('--save-every', default=100, show_default=True)
@click.option('--cuda/--no-cuda', default=True)
@click.option('--max-restarts', default=3, show_default=True)
@click.option('--hook', is_flag=True)
def train_at(workdir, save_every, cuda, max_restarts, hook):
    workdir = Path(workdir).resolve()
    if hook:
        sys.path.append(str(workdir))
        import dlqmc_hook  # noqa: F401
    state_file = workdir / 'state.pt'
    state = torch.load(state_file) if state_file.is_file() else None
    wf, params = wf_from_file(workdir / 'param.toml')
    if cuda:
        wf.cuda()
    for attempt in range(max_restarts + 1):
        try:
            train(
                wf,
                workdir=workdir,
                state=state,
                save_every=save_every,
                **params.get('train_kwargs', {}),
            )
        except TrainingCrash as e:
            log.warning(f'Caught exception: {e.__cause__!r}')
            if attempt == max_restarts:
                log.error('Maximum number of restarts reached')
                break
            state = e.state
            if state:
                log.warning(f'Restarting from step {state["step"]}')
            else:
                log.warning('Restarting from beginning')
        else:
            break


@cli.command('evaluate')
@click.argument('workdir', type=click.Path(exists=True))
@click.option('--cuda/--no-cuda', default=True)
@click.option('--store-steps/--no-store-steps', default=False)
@click.option('--hook', is_flag=True)
def evaluate_at(workdir, cuda, store_steps, hook):
    workdir = Path(workdir).resolve()
    if hook:
        sys.path.append(str(workdir))
        import dlqmc_hook  # noqa: F401
    state = torch.load(workdir / 'state.pt', map_location=None if cuda else 'cpu')
    for _ in range(20):
        try:
            wf, params = wf_from_file(workdir / 'param.toml', state)
        except RuntimeError as exp:
            if 'size mismatch for conf_coeff.weight' not in exp.args[0]:
                raise
        else:
            break
    if cuda:
        wf.cuda()
    evaluate(
        wf,
        store_steps=store_steps,
        workdir=workdir,
        **params.get('evaluate_kwargs', {}),
    )


def get_status(path):
    path = Path(path)
    with path.open() as f:
        lines = f.readlines()
    line = ''
    restarts = 0
    for l in lines:
        if 'E=' in l:
            line = l
        elif 'Restarting' in l:
            restarts += 1
    modtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(path.stat().st_mtime),)
    return {'modtime': modtime, 'restarts': restarts, 'line': line.strip()}


def get_status_multi(paths):
    for path in sorted(paths):
        p = Path(path)
        yield {'path': p.parent, **get_status(p)}


@cli.command()
@click.argument('paths', nargs=-1, type=click.Path(exists=True, dir_okay=False))
def status(paths):
    for x in get_status_multi(paths):
        click.echo('{line} -- {modtime}, restarts: {restarts} | {path}'.format_map(x))


@cli.command()
@click.argument('basedir', type=click.Path(exists=False))
@click.argument('HF', type=float)
@click.argument('exact', type=float)
@click.option('--fractions', default='0,90,99,100', type=str)
@click.option('--steps', '-n', default=2_000, type=int)
def draw_hlines(basedir, hf, exact, fractions, steps):
    basedir = Path(basedir)
    fractions = [float(x) / 100 for x in fractions.split(',')]
    for fraction in fractions:
        value = hf + fraction * (exact - hf)
        workdir = basedir / f'line-{value:.3f}'
        with SummaryWriter(log_dir=workdir, flush_secs=15, purge_step=0) as writer:
            for step in range(steps):
                writer.add_scalar('E_loc_loss/mean', value, step)
                writer.add_scalar('E_loc/mean', value, step)
