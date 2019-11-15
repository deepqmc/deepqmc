from copy import deepcopy
from functools import partial
from importlib import resources
from pathlib import Path

import click
import toml
import torch
from pyscf import gto, scf
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange

from .ansatz import OmniSchnet
from .fit import LossWeightedLogProb, batched_sampler, fit_wfnet
from .geom import geomdb
from .nn import PauliNet
from .sampling import LangevinSampler, rand_from_mf
from .utils import AttrDict

DEFAULTS = toml.loads(
    resources.read_text('dlqmc', 'default-params.toml'), _dict=AttrDict
)


def get_default_params():
    return deepcopy(DEFAULTS)


def model(*, geomname, basis, charge, spin, pauli_kwargs, omni_kwargs):
    mol = gto.M(
        atom=geomdb[geomname].as_pyscf(),
        unit='bohr',
        basis=basis,
        charge=charge,
        spin=spin,
        cart=True,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    wfnet = PauliNet.from_pyscf(
        mf,
        omni_factory=partial(OmniSchnet, **omni_kwargs),
        cusp_correction=True,
        cusp_electrons=True,
        **pauli_kwargs,
    )
    return wfnet, mf


def train(
    wfnet,
    mf,
    *,
    cwd=None,
    state=None,
    save_every=None,
    clip_outliers,
    cuda,
    learning_rate,
    n_steps,
    sampler_size,
    sampler_kwargs,
):
    batched_sampler_kwargs = sampler_kwargs.copy()
    tau = batched_sampler_kwargs.pop('tau')
    rs = rand_from_mf(mf, sampler_size)
    if cuda:
        rs = rs.cuda()
        wfnet.cuda()
    opt = torch.optim.AdamW(wfnet.parameters(), lr=learning_rate)
    if state:
        init_step = state['step'] + 1
        wfnet.load_state_dict(state['wfnet'])
        opt.load_state_dict(state['opt'])
    else:
        init_step = 0
    with SummaryWriter(log_dir=cwd, flush_secs=15, purge_step=init_step - 1) as writer:
        for step in fit_wfnet(
            wfnet,
            LossWeightedLogProb(),
            opt,
            batched_sampler(
                LangevinSampler(wfnet, rs, tau=tau, n_first_certain=3),
                range_sampling=partial(trange, desc='sampling', leave=False),
                **batched_sampler_kwargs,
            ),
            trange(
                init_step, n_steps, initial=init_step, total=n_steps, desc='training'
            ),
            writer=writer,
            clip_outliers=clip_outliers,
        ):
            if cwd and save_every and (step + 1) % save_every == 0:
                state = {
                    'step': step,
                    'wfnet': wfnet.state_dict(),
                    'opt': opt.state_dict(),
                }
                torch.save(state, Path(cwd) / f'state-{step:05d}.pt')


@click.command('train')
@click.argument('path', type=click.Path(exists=True, dir_okay=False))
@click.option('--state', type=click.Path(dir_okay=False))
@click.option('--save-every', default=100, show_default=True)
def train_from_file(path, state, save_every):
    path = Path(path)
    with path.open() as f:
        params = toml.load(f, _dict=AttrDict)
    wfnet, mf = model(**params.model_kwargs)
    if state:
        state = Path(state)
        if state.is_file():
            state = torch.load(state)
    train(
        wfnet,
        mf,
        cwd=path.parent,
        state=state,
        save_every=save_every,
        **params.train_kwargs,
    )
