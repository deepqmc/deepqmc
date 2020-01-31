from functools import partial
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange

from .fit import LossEnergy, fit_wf
from .sampling import LangevinSampler

__version__ = '0.1.0'
__all__ = ['train']


def train(
    wf,
    *,
    n_steps=10_000,
    batch_size=10_000,
    epoch_size=100,
    optimizer='AdamW',
    learning_rate=0.01,
    lr_scheduler='inverse',
    decay_rate=200,
    sampler_kwargs=None,
    fit_kwargs=None,
    cwd=None,
    save_every=None,
    state=None,
):
    r"""Train a wave function model.

    This is the main top-level entry point of the DeepQMC package. This function
    initializes a :class:`~deepqmc.sampling.LangevinSampler`, an optimizer, a
    learning rate scheduler, optionally restores a previously stored training
    state, sets up a Tensorboard writer, and calls :func:`~deepqmc.fit.fit_wf`.

    Args:
        wf (:class:`~deepqmc.wf.WaveFunction`): wave function model to be trained
        n_steps (int): number of optimization steps
        batch_size (int): number of samples used in a single step
        epoch_size (int): number of steps between sampling from the wave function
        optimizer (str): name of the optimizer from :mod:`torch.optim`
        learning_rate (float): learning rate for gradient-descent optimizers
        lr_scheduler (str): name of the learning rate scheduling scheme

            - :data:`None` -- no learning rate scheduling
            - ``'inverse'`` -- :math:`\mathrm{lr}(n):=1/(1+n/r)`, where *r*
              is decay rate
        decay_rate (int): *r*, decay rate for learning rate scheduling
        sampler_kwargs (dict): arguments passed to
            :class:`~deepqmc.sampling.LangevinSampler`
        fit_kwargs (dict): arguments passed to :func:`~deepqmc.fit.fit_wf`
        cwd (str): path where to store Tensorboard event file and intermediate
            parameter states
        save_every (int): number of steps between storing current parameter state
        state (dict): restore optimizer and scheduler states from a stored state
    """
    opt = getattr(torch.optim, optimizer)(wf.parameters(), lr=learning_rate)
    if lr_scheduler == 'inverse':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda n: 1 / (1 + n / decay_rate)
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
        writer.add_text(
            'hyperparameters',
            ''.join(f'**{key}** = {val}  \n' for key, val in locals().items()),
        )
        sampler = LangevinSampler.from_mf(wf, writer=writer, **(sampler_kwargs or {}))
        for step in fit_wf(
            wf,
            LossEnergy(),
            opt,
            sampler.iter_batches(
                batch_size=batch_size,
                epoch_size=epoch_size,
                range=partial(trange, desc='sampling', leave=False),
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
                    'wf': wf.state_dict(),
                    'opt': opt.state_dict(),
                }
                if scheduler:
                    state['scheduler'] = scheduler.state_dict()
                torch.save(state, Path(cwd) / f'state-{step:05d}.pt')
