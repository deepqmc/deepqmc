import inspect
import logging
import sys
from pathlib import Path

import click
import tomlkit
from tomlkit.items import Comment, Trivia
from tqdm import tqdm

from .errors import TrainingCrash
from .evaluate import evaluate
from .fit import fit_wf
from .io import wf_from_file
from .sampling import LangevinSampler, sample_wf
from .train import train
from .wf import ANSATZES

__all__ = ()

log = logging.getLogger(__name__)

DEEPQMC_DEFAULTS = {
    (train, 'sampler_kwargs'): LangevinSampler.from_wf,
    (train, 'fit_kwargs'): fit_wf,
    (train, 'optimizer_kwargs'): True,
    (train, 'lr_scheduler_kwargs'): True,
    (LangevinSampler.from_wf, 'kwargs'): LangevinSampler,
    (evaluate, 'sampler_kwargs'): (
        LangevinSampler.from_wf,
        [('n_decorrelate', 4), 'n_discard', 'sample_size'],
    ),
    (evaluate, 'sample_kwargs'): sample_wf,
}


def _get_subkwargs(func, name, mapping):
    target = mapping[func, name]
    target, override = target if isinstance(target, tuple) else (target, [])
    if isinstance(target, dict):
        sub_kwargs = {k: collect_kwarg_defaults(v, mapping) for k, v in target.items()}
    else:
        sub_kwargs = collect_kwarg_defaults(target, mapping)
    for x in override:
        if isinstance(x, tuple):
            key, val = x
            sub_kwargs[key] = val
        else:
            del sub_kwargs[x]
    return sub_kwargs


def collect_kwarg_defaults(func, mapping):
    kwargs = tomlkit.table()
    for p in inspect.signature(func).parameters.values():
        if p.name == 'kwargs':
            assert p.default is p.empty
            assert p.kind is inspect.Parameter.VAR_KEYWORD
            sub_kwargs = _get_subkwargs(func, 'kwargs', mapping)
            for item in sub_kwargs.value.body:
                kwargs.add(*item)
        elif p.name.endswith('_kwargs'):
            if mapping.get((func, p.name)) is True:
                kwargs[p.name] = p.default
            else:
                assert p.default is None
                assert p.kind is inspect.Parameter.KEYWORD_ONLY
                sub_kwargs = _get_subkwargs(func, p.name, mapping)
                kwargs[p.name] = sub_kwargs
        elif p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            assert p.default in (p.empty, p.default)
        else:
            assert p.kind is inspect.Parameter.KEYWORD_ONLY
            if p.default is None:
                kwargs.add(Comment(Trivia(comment=f'#: {p.name} = ...')))
            else:
                try:
                    kwargs[p.name] = p.default
                except ValueError:
                    print(func, p.name, p.kind, p.default)
                    raise
    return kwargs


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
@click.option('-v', '--verbose', count=True, help='Increase verbosity.')
@click.option('-q', '--quiet', is_flag=True, help='Suppres all output.')
def cli(verbose, quiet):  # noqa: D403
    """DeepQMC runs quantum Monte Carlo with deep neural networks."""
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
@click.option(
    '--commented', '-c', is_flag=True, help='Comment out all hyperparameters.'
)
def defaults(commented):
    """Print all hyperparameters and their default values.

    The hyperparameters are printed in the TOML format that is expected by other
    deepqmc commands.
    """
    table = tomlkit.table()
    table['train_kwargs'] = collect_kwarg_defaults(train, DEEPQMC_DEFAULTS)
    table['evaluate_kwargs'] = collect_kwarg_defaults(evaluate, DEEPQMC_DEFAULTS)
    for label, ansatz in ANSATZES.items():
        table[f'{label}_kwargs'] = collect_kwarg_defaults(ansatz.entry, ansatz.defaults)
    lines = tomlkit.dumps(table).split('\n')
    if commented:
        lines = ['# ' + l if ' = ' in l and l[0] != '#' else l for l in lines]
    click.echo('\n'.join(lines), nl=False)


@cli.command('train')
@click.argument('workdir', type=click.Path(exists=True))
@click.option(
    '--save-every',
    default=100,
    show_default=True,
    help='Frequency in steps of saving the curent state of the optimization.',
)
@click.option(
    '--cuda/--no-cuda',
    default=True,
    show_default=True,
    help='Toggle training on a GPU.',
)
@click.option(
    '--max-restarts',
    default=3,
    show_default=True,
    help='Maximum number of attempted restarts before aborting.',
)
@click.option('--hook', is_flag=True, help='Import a deepqmc hook from WORKDIR.')
def train_at(workdir, save_every, cuda, max_restarts, hook):
    """Train an ansatz with variational quantum Monte Carlo.

    The calculation details must be specified in a "param.toml" file in WORKDIR,
    which must contain at least the keywords "system" and "ansatz", and
    optionally any keywords printed by the "defaults" command.
    """
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
@click.option(
    '--cuda/--no-cuda',
    default=True,
    show_default=True,
    help='Toggle training on a GPU.',
)
@click.option(
    '--store-steps/--no-store-steps',
    default=False,
    show_default=True,
    help='Toggle storing of individual sampling steps.',
)
@click.option('--hook', is_flag=True)
def evaluate_at(workdir, cuda, store_steps, hook):
    """Estimate total energy of an ansatz via Monte Carlo sampling.

    The calculation details must be specified in a "param.toml" file in WORKDIR,
    which must contain at least the keywords "system" and "ansatz", and
    optionally any keywords printed by the "defaults" command. The wave function
    ansatz must be stored in a "state.pt" file in WORKDIR, which was generated
    with the "train" command.
    """
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
