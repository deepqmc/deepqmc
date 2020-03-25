from functools import partial
from itertools import count
from pathlib import Path

import tables
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange

from .fit import LossEnergy, fit_wf
from .sampling import LangevinSampler, sample_wf

__version__ = '0.1.0'
__all__ = ['train']


def train(  # noqa: C901
    wf,
    workdir=None,
    save_every=None,
    state=None,
    _optimizer=None,
    _sampler_state=None,
    *,
    n_steps=10_000,
    batch_size=10_000,
    epoch_size=100,
    optimizer='AdamW',
    learning_rate=0.01,
    lr_scheduler='inverse',
    decay_rate=200,
    equilibrate=True,
    fit_kwargs=None,
    sampler_kwargs=None,
    reset_memory=False,
    reset_tolerance=0.005,
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
        workdir (str): path where to store Tensorboard event file, intermediate
            parameter states, and HDF5 file with the fit trajectory
        save_every (int): number of steps between storing current parameter state
        state (dict): restore optimizer and scheduler states from a stored state
    """
    opt = _optimizer or getattr(torch.optim, optimizer)(
        wf.parameters(), lr=learning_rate
    )
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
    if workdir:
        workdir = Path(workdir)
        writer = SummaryWriter(log_dir=workdir, flush_secs=15, purge_step=init_step - 1)
        writer.add_text(
            'hyperparameters',
            ''.join(f'**{key}** = {val}  \n' for key, val in locals().items()),
        )
        chkpts_dir = workdir / 'chkpts'
        chkpts_dir.mkdir(exist_ok=True)
        h5file = tables.open_file(workdir / 'fit.h5', 'a')
        if 'fit' not in h5file.root:
            desc = {
                label: tables.Float32Col(batch_size)
                for label in ['E_loc', 'E_loc_loss', 'log_psis', 'sign_psis', 'log_ws']
            }
            table = h5file.create_table('/', 'fit', desc)
        else:
            table = h5file.root['fit']
            table.remove_rows(init_step)
    else:
        writer = None
    sampler = LangevinSampler.from_mf(wf, writer=writer, **(sampler_kwargs or {}))
    if _sampler_state:
        sampler.load_state_dict(_sampler_state)
    steps = trange(
        init_step, n_steps, initial=init_step, total=n_steps, desc='training'
    )
    restart_from = False
    if reset_memory:
        assert workdir
    try:
        if equilibrate:
            with tqdm(count(), desc='equilibrating') as eq_steps:
                next(sample_wf(wf, sampler.iter_with_info(), eq_steps))
            steps.unpause()
        for step, energy in fit_wf(
            wf,
            LossEnergy(),
            opt,
            sampler.iter_batches(
                batch_size=batch_size,
                epoch_size=epoch_size,
                range=partial(trange, desc='sampling', leave=False),
            ),
            steps,
            log_dict=table.row if workdir else None,
            writer=writer,
            **(fit_kwargs or {}),
        ):
            steps.set_postfix(E=f'{energy:S}')
            if scheduler:
                scheduler.step()
            if workdir:
                table.row.append()
                table.flush()
                if save_every and (step + 1) % save_every == 0:
                    state = {
                        'step': step,
                        'wf': wf.state_dict(),
                        'opt': opt.state_dict(),
                    }
                    if scheduler:
                        state['scheduler'] = scheduler.state_dict()
                    torch.save(state, chkpts_dir / f'state-{step:05d}.pt')
            if reset_memory and step > reset_memory:
                energy_memory = torch.from_numpy(
                    h5file.root.fit.col('E_loc')[-reset_memory:].mean(axis=-1)
                )
                mean_old, mean_new = map(
                    lambda x: x.sort()[0][
                        reset_memory // 10 : -reset_memory // 10 - 1
                    ].mean(),
                    (
                        energy_memory[: reset_memory // 2],
                        energy_memory[reset_memory // 2 :],
                    ),
                )
                if (mean_new - (1 - reset_tolerance) * mean_old) / (
                    mean_new + mean_old
                ) < 0:
                    restart_from = torch.load(
                        chkpts_dir
                        / f'state-{int((step-reset_memory/2)//save_every*save_every-1):05d}.pt'
                    )
                    print(
                        f'restore state at step {int((step-reset_memory/2)//save_every*save_every-1)}'
                    )
                    break
    finally:
        steps.close()
        if workdir:
            writer.close()
            h5file.close()
        if restart_from:
            wf.load_state_dict(restart_from['wf']),
            train(
                wf,
                workdir=workdir,
                save_every=save_every,
                state=restart_from,
                n_steps=n_steps,
                batch_size=batch_size,
                epoch_size=epoch_size,
                optimizer=optimizer,
                learning_rate=learning_rate,
                lr_scheduler=lr_scheduler,
                decay_rate=decay_rate,
                equilibrate=True,
                fit_kwargs=fit_kwargs,
                sampler_kwargs=sampler_kwargs,
                reset_memory=reset_memory,
                reset_tolerance=reset_tolerance,
            )
