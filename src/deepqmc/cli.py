import logging
import sys
from pathlib import Path

import click
import tomlkit
from tqdm import tqdm

from .defaults import DEEPQMC_MAPPING, collect_kwarg_defaults
from .errors import TrainingCrash
from .evaluate import evaluate
from .io import wf_from_file
from .train import train
from .wf import PauliNet

__all__ = ()

log = logging.getLogger(__name__)


class TqdmStream:
    def write(self, msg):
        tqdm.write(msg, end='')


class CLI(click.Group):
    def list_commands(self, ctx):
        return self.commands.keys()

    def get_command(self, ctx, name):
        if name.startswith('extra:'):
            from .extra import cli as extra_cli

            name = name.split(':', 1)[1]
            for attr in dir(extra_cli):
                cmd = getattr(extra_cli, attr)
                if isinstance(cmd, click.core.Command) and cmd.name == name:
                    return cmd
        return super().get_command(ctx, name)


@click.group(cls=CLI)
@click.option('-v', '--verbose', count=True)
@click.option('-q', '--quiet', is_flag=True)
def cli(verbose, quiet):
    assert not (quiet and verbose)
    logging.basicConfig(
        style='{',
        format='[{asctime}.{msecs:03.0f}] {levelname}:{name}: {message}',
        datefmt='%H:%M:%S',
        stream=TqdmStream(),
    )
    if quiet:
        level = logging.ERROR
    else:
        level = [logging.WARNING, logging.INFO, logging.DEBUG][verbose]
    logging.getLogger('deepqmc').setLevel(level)


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
        log.info('Importing a dlqmc hook')
        sys.path.append(str(workdir))
        import dlqmc_hook  # noqa: F401
    state = None
    for attempt in range(max_restarts + 1):
        log.info('Initializing a new wave function')
        wf, params, state_from_file = wf_from_file(workdir)
        state = state or state_from_file
        if cuda:
            log.info('Moving to GPU...')
            wf.cuda()
            log.info('Moved to GPU')
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
            state = e.state
            if attempt == max_restarts:
                log.error('Maximum number of restarts reached')
                break
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
    wf, params, state = wf_from_file(workdir)
    if state:
        wf.load_state_dict(state['wf'])
    if cuda:
        wf.cuda()
    evaluate(
        wf,
        store_steps=store_steps,
        workdir=workdir,
        **params.get('evaluate_kwargs', {}),
    )
