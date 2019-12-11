from functools import partial
from pathlib import Path

import click
import toml
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange

from . import Molecule
from .fit import LossWeightedLogProb, batched_sampler, fit_wfnet
from .sampling import LangevinSampler, rand_from_mf
from .wf import PauliNet


def train(
    wfnet,
    *,
    cwd=None,
    state=None,
    save_every=None,
    cuda=True,
    learning_rate=0.01,
    n_steps=10_000,
    sampler_size=2_000,
    lr_scheduler='inverse',
    decay_rate=200,
    optimizer='AdamW',
    sampler_kwargs=None,
    batched_sampler_kwargs=None,
    fit_kwargs=None,
):
    rs = rand_from_mf(wfnet.mf, sampler_size)
    if cuda:
        rs = rs.cuda()
        wfnet.cuda()
    opt = getattr(torch.optim, optimizer)(wfnet.parameters(), lr=learning_rate)
    if lr_scheduler == 'inverse':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda t: 1 / (1 + t / decay_rate)
        )
    else:
        scheduler = None
    if state:
        init_step = state['step'] + 1
        opt.load_state_dict(state['opt'])
        if scheduler:
            scheduler.load_state_dict(state['scheduler'])
    else:
        init_step = 0
    with SummaryWriter(log_dir=cwd, flush_secs=15, purge_step=init_step - 1) as writer:
        for step in fit_wfnet(
            wfnet,
            LossWeightedLogProb(),
            opt,
            batched_sampler(
                LangevinSampler(wfnet, rs, **(sampler_kwargs or {})),
                range_sampling=partial(trange, desc='sampling', leave=False),
                **(batched_sampler_kwargs or {}),
            ),
            trange(
                init_step, n_steps, initial=init_step, total=n_steps, desc='training'
            ),
            writer=writer,
            **(fit_kwargs or {}),
        ):
            if scheduler:
                scheduler.step()
            if cwd and save_every and (step + 1) % save_every == 0:
                state = {
                    'step': step,
                    'wfnet': wfnet.state_dict(),
                    'opt': opt.state_dict(),
                }
                if scheduler:
                    state['scheduler'] = scheduler.state_dict()
                torch.save(state, Path(cwd) / f'state-{step:05d}.pt')


def state_from_file(path):
    return torch.load(path) if path and Path(path).is_file() else None


def model_from_file(path, state=None):
    param = toml.loads(Path(path).read_text())
    system = param.pop('system')
    if isinstance(system, str):
        system = {'name': system}
    mol = Molecule.from_name(**system)
    wfnet = PauliNet.from_hf(mol, **param['model_kwargs'])
    if state:
        wfnet.load_state_dict(state['wfnet'])
    return wfnet, param


@click.command('train')
@click.argument('path', type=click.Path(exists=True, dir_okay=False))
@click.option('--state', type=click.Path(dir_okay=False))
@click.option('--save-every', default=100, show_default=True)
def train_from_file(path, state, save_every):
    state = state_from_file(state)
    wfnet, param = model_from_file(path, state)
    train(
        wfnet,
        cwd=Path(path).parent,
        state=state,
        save_every=save_every,
        **param['train_kwargs'],
    )
