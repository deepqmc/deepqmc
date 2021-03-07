import logging
import sys
import time
import traceback
from itertools import count
from math import inf
from pathlib import Path

import click
import torch
from torch.utils.tensorboard import SummaryWriter

from ..errors import TrainingCrash
from ..io import wf_from_file
from ..train import train

__all__ = ()

log = logging.getLogger(__name__)


@click.command('train-multi')  # noqa: C901
@click.argument('workdir', type=click.Path(exists=True))
@click.argument('respawn', type=int)
@click.option('--multi-part', default=0)
@click.option('--timeout-short', default=30 * 60)
@click.option('--timeout-long', default=2 * 60 * 60)
@click.option('--check-interval', default=30)
@click.option('--cuda/--no-cuda', default=True)
@click.option('--max-restarts', default=3, show_default=True)
@click.option('--hook', is_flag=True)
def train_multi_at(  # noqa: C901
    workdir,
    respawn,
    multi_part,
    timeout_short,
    timeout_long,
    check_interval,
    cuda,
    max_restarts,
    hook,
):
    workdir = Path(workdir).resolve()
    rank = int(workdir.parts[::-1][multi_part])
    if hook:
        log.info('Importing a dlqmc hook')
        sys.path.append(str(workdir))
        import dlqmc_hook  # noqa: F401
    state = None
    chkpts = None
    for cycle in count():
        end_step = (cycle + 1) * respawn
        for attempt in range(max_restarts + 1):
            log.info('Initializing a new wave function')
            wf, params, state_from_file = wf_from_file(workdir)
            state = state or state_from_file
            if cuda:
                log.info('Moving to GPU...')
                wf.cuda()
                log.info('Moved to GPU')
            try:
                interrupted = train(
                    wf,
                    workdir=workdir,
                    state=state,
                    chkpts=chkpts,
                    raise_blowup=False,
                    save_every=respawn,
                    return_every=respawn,
                    blowup_threshold=inf,
                    **params.get('train_kwargs', {}),
                )
            except TrainingCrash as e:
                log.warning(f'Training crash in cycle {cycle}, attempt {attempt}')
                log.warning('\n' + traceback.format_exc().strip())
                state, chkpts = e.state, e.chkpts
                if (
                    attempt == max_restarts
                    or not state
                    or state['step'] <= cycle * respawn
                ):
                    log.warning('Aborting cycle')
                    (workdir / 'chkpts' / f'state-{end_step:05d}.STOP').touch()
                    interrupted = True
                    break
                if state:
                    log.warning(f'Restarting from step {state["step"]}')
                else:
                    log.warning('Restarting from beginning')
            else:
                break
        if not interrupted:
            return
        start = time.time()
        while True:
            now = time.time()
            root = workdir.parents[multi_part]
            stem = ('*',) + workdir.parts[::-1][:multi_part]
            root.glob('/'.join(stem + ('param.toml',)))
            n_tasks = len(list(root.glob('/'.join(stem + ('param.toml',)))))
            all_states = {
                int(p.parts[-3 - multi_part]): p
                for p in root.glob(
                    '/'.join(stem + (f'chkpts/state-{end_step:05d}.pt',))
                )
            }
            all_stops = {
                int(p.parts[-3 - multi_part]): None
                for p in root.glob(
                    '/'.join(stem + (f'chkpts/state-{end_step:05d}.STOP',))
                )
            }
            all_states = {**all_states, **all_stops}
            n_all_states = len(all_states)
            log.info(f'{n_all_states}/{n_tasks} states ready')
            if n_all_states < n_tasks / 2 and now - start < timeout_long:
                log.info('Missing >1/2 states and long timeout not up, waiting...')
                time.sleep(check_interval)
                continue
            if n_all_states < n_tasks and now - start < timeout_short:
                log.info('Missing some states and short timeout not up, waiting...')
                time.sleep(check_interval)
                continue
            all_states = [(p, torch.load(p)) for p in all_states.values() if p]
            log.info(f'Have {len(all_states)} states for respawning')
            if not all_states:
                log.error('No states for respawning, abort')
                return
            all_states.sort(key=lambda x: x[1]['monitor'].mean_of('mean_slow'))
            all_states = all_states[: n_tasks // 2]
            path, state = all_states[rank % len(all_states)]
            log.info(f'Respawning from {path}')
            break


def get_status(path):
    path = Path(path).resolve()
    with path.open() as f:
        lines = f.readlines()
    line = ''
    restarts = 0
    for l in lines:
        if 'E=' in l or 'energy =' in l:
            line = l
        elif 'Restarting' in l:
            restarts += 1
    modtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(path.stat().st_mtime))
    return {'modtime': modtime, 'restarts': restarts, 'line': line.strip()}


def get_status_multi(paths):
    for path in sorted(paths):
        p = Path(path)
        yield {'path': p.parent, **get_status(p)}


@click.command()
@click.argument('paths', nargs=-1, type=click.Path(exists=True, dir_okay=False))
def status(paths):
    for x in get_status_multi(paths):
        click.echo('{line} -- {modtime}, restarts: {restarts} | {path}'.format_map(x))


@click.command()
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
