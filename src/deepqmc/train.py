from functools import partial
from itertools import count
from pathlib import Path

import h5py
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange

from .errors import NanGradients, NanLoss, TrainingBlowup, TrainingCrash
from .fit import LossEnergy, fit_wf
from .plugins import PLUGINS
from .sampling import LangevinSampler, sample_wf
from .utils import H5LogTable

__version__ = '0.1.0'
__all__ = ['train']

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
    ewm_decay_rate=20,
    outlier_sigma=3,
    max_outliers=3,
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
    if state:
        init_step = state['step'] + 1
        opt.load_state_dict(state['opt'])
        if scheduler:
            scheduler.load_state_dict(state['scheduler'])
        ewm_mean, ewm_std = state['ewm']
    else:
        init_step = 0
        ewm_mean = None
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
        columns = {
            label: (batch_size,)
            for label in ['E_loc', 'E_loc_loss', 'log_psis', 'sign_psis', 'log_ws']
        }
        columns['learning_rate'] = ()
        table = H5LogTable(h5file, columns)
        table.resize(init_step)
    else:
        writer = None
    if 'sampler_factory' in PLUGINS:
        sampler = PLUGINS['sampler_factory'](wf, writer=writer)
    else:
        sampler = LangevinSampler.from_mf(wf, writer=writer, **(sampler_kwargs or {}))
    steps = trange(
        init_step, n_steps, initial=init_step, total=n_steps, desc='training'
    )
    chkpts = []
    outlier_count = 0
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
            ewm_decay = 1 - 1 / (2 + step / ewm_decay_rate)
            if ewm_mean is None:
                ewm_mean, ewm_std = energy, energy.std_dev
            elif abs(energy - ewm_mean) < outlier_sigma * ewm_std:
                ewm_mean = (1 - ewm_decay) * energy + ewm_decay * ewm_mean
                ewm_std = (1 - ewm_decay) * energy.std_dev + ewm_decay * ewm_std
                outlier_count = 0
            elif outlier_count < max_outliers:
                outlier_count += 1
            else:
                outlier_sigma = abs(energy - ewm_mean) / ewm_std
                raise TrainingBlowup(f'Outlier sigma: {outlier_sigma}')
            steps.set_postfix(E=f'{ewm_mean:S} (s={ewm_std:.3f})')
            if scheduler:
                scheduler.step()
            if workdir:
                h5file.flush()
                if save_every and (step + 1) % save_every == 0:
                    state = {
                        'step': step,
                        'wf': wf.state_dict(),
                        'opt': opt.state_dict(),
                        'ewm': (ewm_mean, ewm_std),
                    }
                    if scheduler:
                        state['scheduler'] = scheduler.state_dict()
                    state_file = chkpts_dir / f'state-{step + 1:05d}.pt'
                    chkpts.append((step, state_file))
                    torch.save(state, state_file)
    except (NanLoss, NanGradients, TrainingBlowup) as e:
        raise TrainingCrash(step, chkpts) from e
    except RuntimeError as e:
        if 'svd_cuda: the updating process of SBDSDC did not converge' in e.args[0]:
            raise TrainingCrash(step, chkpts) from e
    finally:
        steps.close()
        if workdir:
            writer.close()
            h5file.close()
