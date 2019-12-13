from functools import partial
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange

from .fit import LossWeightedLogProb, fit_wfnet
from .sampling import LangevinSampler


def train(
    wfnet,
    *,
    cwd=None,
    state=None,
    save_every=None,
    cuda=True,
    learning_rate=0.01,
    n_steps=10_000,
    lr_scheduler='inverse',
    decay_rate=200,
    epoch_size=100,
    batch_size=10_000,
    optimizer='AdamW',
    sampler_kwargs=None,
    fit_kwargs=None,
):
    if cuda:
        wfnet.cuda()
    sampler = LangevinSampler.from_mf(wfnet, cuda=cuda, **(sampler_kwargs or {}))
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
                    'wfnet': wfnet.state_dict(),
                    'opt': opt.state_dict(),
                }
                if scheduler:
                    state['scheduler'] = scheduler.state_dict()
                torch.save(state, Path(cwd) / f'state-{step:05d}.pt')
