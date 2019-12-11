from pathlib import Path

import click
import toml
import torch

from . import Molecule, train
from .wf import PauliNet


def wfnet_from_file(path, state=None):
    param = toml.loads(Path(path).read_text())
    system = param.pop('system')
    if isinstance(system, str):
        system = {'name': system}
    mol = Molecule.from_name(**system)
    wfnet = PauliNet.from_hf(mol, **param['model_kwargs'])
    if state:
        wfnet.load_state_dict(state['wfnet'])
    return wfnet, param


@click.command()
@click.argument('path', type=click.Path(exists=True, dir_okay=False))
@click.option('--state', type=click.Path(dir_okay=False))
@click.option('--save-every', default=100, show_default=True)
def train_from_file(path, state, save_every):
    state = torch.load(state) if state and Path(state).is_file() else None
    wfnet, param = wfnet_from_file(path, state)
    train(
        wfnet,
        cwd=Path(path).parent,
        state=state,
        save_every=save_every,
        **param['train_kwargs'],
    )
