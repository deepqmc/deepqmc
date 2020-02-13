from pathlib import Path

import click
import toml
import torch

from . import Molecule, train
from .wf import PauliNet

__all__ = ()


@click.group()
def cli():
    pass


def wf_from_file(path, state=None):
    param = toml.loads(Path(path).read_text())
    system = param.pop('system')
    if isinstance(system, str):
        system = {'name': system}
    mol = Molecule.from_name(**system)
    wf = PauliNet.from_hf(mol, **param['model_kwargs'])
    if state:
        wf.load_state_dict(state['wf'])
    return wf, param


@cli.command('train')
@click.argument('path', type=click.Path(exists=True, dir_okay=False))
@click.option('--state', type=click.Path(dir_okay=False))
@click.option('--save-every', default=100, show_default=True)
@click.option('--cuda/--no-cuda')
def train_from_file(path, state, save_every, cuda):
    state = torch.load(state) if state and Path(state).is_file() else None
    wf, param = wf_from_file(path, state)
    if cuda:
        wf.cuda()
    train(
        wf,
        cwd=Path(path).parent,
        state=state,
        save_every=save_every,
        **param['train_kwargs'],
    )
