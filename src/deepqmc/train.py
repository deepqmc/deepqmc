import logging
from copy import deepcopy
from functools import partial
from itertools import count
from math import inf
from pathlib import Path

import h5py
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange

from .errors import NanGradients, NanLoss, TrainingBlowup, TrainingCrash
from .ewm import EWMElocMonitor
from .fit import LossEnergy, fit_wf
from .plugins import PLUGINS
from .sampling import LangevinSampler, sample_wf
from .utils import H5LogTable

__version__ = '0.1.0'
__all__ = ['train']

log = logging.getLogger(__name__)

OPTIMIZER_KWARGS = {
    'Adam': {'betas': [0.9, 0.9]},
    'AdamW': {'betas': [0.9, 0.9], 'weight_decay': 0.01},
}
SCHEDULER_KWARGS = {
    'CyclicLR': {
        'base_lr': 1e-4,
        'max_lr': 10e-3,
        'step_size_up': 250,
        'mode': 'triangular2',
        'cycle_momentum': False,
    },
    'OneCycleLR': {
        'max_lr': 5e-3,
        'total_steps': 5_000,
        'pct_start': 0.075,
        'anneal_strategy': 'linear',
    },
    'inverse': {'decay_rate': 200},
    'scan': {'eq_steps': 100, 'start': 0.1, 'rate': 1.05},
}


def train(  # noqa: C901
    wf,
    workdir=None,
    save_every=None,
    state=None,
    min_rewind=10,
    blowup_threshold=0.5,
    return_every=None,
    *,
    n_steps=10_000,
    batch_size=10_000,
    epoch_size=100,
    optimizer='AdamW',
    learning_rate=0.01,
    optimizer_kwargs=OPTIMIZER_KWARGS,
    lr_scheduler='CyclicLR',
    lr_scheduler_kwargs=SCHEDULER_KWARGS,
    equilibrate=True,
    fit_kwargs=None,
    sampler_kwargs=None,
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
    if 'optimizer_factory' in PLUGINS:
        opt = PLUGINS['optimizer_factory'](wf.parameters())
    else:
        optimizer_kwargs = {
            **OPTIMIZER_KWARGS.get(optimizer, {}),
            **optimizer_kwargs[optimizer],
        }
        opt = getattr(torch.optim, optimizer)(
            wf.parameters(), lr=learning_rate, **optimizer_kwargs
        )
    if 'scheduler_factory' in PLUGINS:
        scheduler = PLUGINS['scheduler_factory'](opt)
    elif lr_scheduler:
        scheduler_kwargs = {
            **SCHEDULER_KWARGS.get(lr_scheduler, {}),
            **lr_scheduler_kwargs[lr_scheduler],
        }
        if lr_scheduler[0].islower():
            if lr_scheduler == 'inverse':

                def lr_lambda(n, decay_rate):
                    return 1 / (1 + n / decay_rate)

            elif lr_scheduler == 'scan':

                def lr_lambda(n, eq_steps, start, rate):
                    return 1.0 if n < eq_steps else start * rate ** (n - eq_steps)

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                opt, partial(lr_lambda, **scheduler_kwargs)
            )
        else:
            scheduler = getattr(torch.optim.lr_scheduler, lr_scheduler)(
                opt, **scheduler_kwargs
            )
    else:
        scheduler = None
    # The convention here is that states/steps are numbered as slices/elements
    # in a Python list. For example, step 0 takes state 0 to state 1. The
    # progress bar really displays states, not steps, as it goes from 0 to
    # n_steps, that is, it goes through n_steps+1 states.
    if state:
        init_step = state['step']
        wf.load_state_dict(state['wf'])
        opt.load_state_dict(state['opt'])
        if scheduler:
            scheduler.load_state_dict(state['scheduler'])
        monitor = state['monitor']
    else:
        init_step = 0
        monitor = EWMElocMonitor()
    if workdir:
        workdir = Path(workdir)
        writer = SummaryWriter(log_dir=workdir, flush_secs=15, purge_step=init_step - 1)
        writer.add_text(
            'hyperparameters',
            ''.join(f'**{key}** = {val}  \n' for key, val in locals().items()),
        )
        chkpts_dir = workdir / 'chkpts'
        chkpts_dir.mkdir(exist_ok=True)
        h5file = h5py.File(workdir / 'fit.h5', 'a', libver='v110')
        h5file.swmr_mode = True
        table = H5LogTable(h5file)
        table.resize(init_step)
        h5file.flush()
    else:
        writer = None
        log_dict = {}
    steps = trange(
        init_step, n_steps, initial=init_step, total=n_steps, desc='training'
    )
    chkpts = []
    step = init_step - 1
    try:
        # this can blowup if the backprop of SVD in determiants fail
        if 'sampler_factory' in PLUGINS:
            sampler = PLUGINS['sampler_factory'](wf, writer=writer)
        else:
            sampler = LangevinSampler.from_mf(
                wf, writer=writer, **(sampler_kwargs or {})
            )
        if equilibrate:
            with tqdm(count(), desc='equilibrating') as eq_steps:
                next(sample_wf(wf, sampler.iter_with_info(), eq_steps))
            steps.unpause()
        for step, _ in fit_wf(
            wf,
            LossEnergy(),
            opt,
            sampler.iter_batches(
                batch_size=batch_size,
                epoch_size=epoch_size,
                range=partial(trange, desc='sampling', leave=False),
            ),
            steps,
            log_dict=table.row if workdir else log_dict,
            writer=writer,
            **(fit_kwargs or {}),
        ):
            # at this point, the wf model and optimizer are already at state step+1
            monitor.update(table['E_loc'][-1] if workdir else log_dict['E_loc'])
            # now monitor is at state `step+1`. if blowup was detected, the
            # blowup is reported to occur at step `step`.
            if monitor.blowup_detection.get('indicator', 0) > blowup_threshold:
                raise TrainingBlowup(repr(monitor.blowup_detection))
            if monitor.energy.std_dev > 0:
                steps.set_postfix(E=f'{monitor.energy:S}')
            state = {
                'step': step + 1,
                'wf': wf.state_dict(),
                'opt': opt.state_dict(),
                'monitor': deepcopy(monitor),
            }
            if scheduler:
                scheduler.step()
                # now scheduler is at state step+1
                state['scheduler'] = scheduler.state_dict()
            chkpts.append((step + 1, state))
            chkpts = chkpts[-100:]
            if workdir:
                table.row['E_ewm'] = monitor.energy.n
                h5file.flush()
                if save_every and (step + 1) % save_every == 0:
                    state_file = chkpts_dir / f'state-{step + 1:05d}.pt'
                    torch.save(state, chkpts_dir / f'state-{step + 1:05d}.pt')
                    log.debug(torch.cuda.memory_summary(abbreviated=True))
            if return_every and (step + 1) % return_every == 0:
                return True
    except (NanLoss, NanGradients, TrainingBlowup, RuntimeError) as e:
        if isinstance(e, RuntimeError):
            if 'the updating process of SBDSDC did not converge' not in e.args[0]:
                raise
        blowup_step = (
            monitor.blowup_detection['init']
            if monitor.blowup_detection and blowup_threshold < inf
            else step
        )
        target_step = blowup_step - min_rewind
        for step, state in reversed(chkpts):
            if step <= target_step:
                raise TrainingCrash(state) from e
        for state_file in sorted(chkpts_dir.glob('state-*.pt'), reverse=True):
            step = int(state_file.stem.split('-')[1])
            if step <= target_step:
                raise TrainingCrash(torch.load(state_file)) from e
        raise TrainingCrash(None) from e
    finally:
        steps.close()
        if workdir:
            writer.close()
            h5file.close()
