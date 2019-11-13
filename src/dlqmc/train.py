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
from .fit import LossWeightedLogProb, fit_wfnet, wfnet_fit_driver
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
    clip_outliers,
    cuda,
    learning_rate,
    n_steps,
    sampler_size,
    tau,
    driver_kwargs,
):
    rs = rand_from_mf(mf, sampler_size)
    if cuda:
        rs = rs.cuda()
        wfnet.cuda()
    sampler = LangevinSampler(wfnet, rs, tau=tau, n_first_certain=3)
    with SummaryWriter(log_dir=cwd, flush_secs=15) as writer:
        for step in fit_wfnet(
            wfnet,
            LossWeightedLogProb(),
            torch.optim.AdamW(wfnet.parameters(), lr=learning_rate),
            wfnet_fit_driver(
                sampler,
                range_sampling=partial(trange, desc='sampling', leave=False),
                **driver_kwargs,
            ),
            trange(n_steps, desc='training'),
            writer=writer,
            clip_outliers=clip_outliers,
        ):
            if cwd and step % 100 == 50:
                torch.save(wfnet, Path(cwd) / f'wfnet-{step}.pt')


@click.command('train')
@click.argument('path', type=click.Path(exists=True, dir_okay=False, resolve_path=True))
def train_from_file(path):
    path = Path(path)
    with path.open() as f:
        params = toml.load(f, _dict=AttrDict)
    net, mf = model(**params.model_kwargs)
    train(net, mf, cwd=path.parent, **params.train_kwargs)
