from itertools import count
from pathlib import Path

import h5py
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from uncertainties import unumpy as unp

from .sampling import LangevinSampler, sample_wf
from .utils import H5LogTable

__version__ = '0.1.0'
__all__ = ['evaluate']


def evaluate(
    wf,
    store_steps=False,
    workdir=None,
    log_dict=None,
    *,
    n_steps=500,
    sample_size=1_000,
    sample_kwargs=None,
    sampler_kwargs=None,
):
    r"""Evaluate a wave function model.

    This is a top-level API function that rigorously evaluates a trained wave
    function model. It initializes a :class:`~deepqmc.sampling.LangevinSampler`,
    sets up a Tensorboard writer, and calls :func:`~deepqmc.sampling.sample_wf`.

    Args:
        wf (:class:`~deepqmc.wf.WaveFunction`): wave function model to be evaluated
        store_steps (bool): whether to store individual sampled electron configuraitons
        workdir (str): path where to store Tensorboard event file and HDF5 file with
            sampling block energies
        log_dict (dict-like): dictionary to store the step data
        n_steps (int): number of sampling steps
        sample_size (int): number of Markov-chain walkers
        sample_kwargs (dict): extra arguments passed to
            :func:`~deepqmc.sampling.sample_wf`
        sampler_kwargs (dict): extra arguments passed to
            :class:`~deepqmc.sampling.LangevinSampler`

    Returns:
        dict: Expectation values with standard errors.
    """
    if workdir:
        workdir = Path(workdir)
        writer = SummaryWriter(log_dir=workdir, flush_secs=15)
        h5file = h5py.File(workdir / 'sample.h5', 'a', libver='v110')
        h5file.swmr_mode = True
        table_blocks = H5LogTable(h5file.require_group('blocks'))
        table_steps = H5LogTable(h5file.require_group('steps'))
    else:
        writer = None
    sampler = LangevinSampler.from_wf(
        wf,
        sample_size=sample_size,
        writer=writer,
        n_discard=0,
        **{'n_decorrelate': 4, **(sampler_kwargs or {})},
    )
    steps = tqdm(count(), desc='equilibrating', disable=None)
    blocks = []
    try:
        for step, energy in sample_wf(
            wf,
            sampler.iter_with_info(),
            steps,
            blocks=blocks,
            log_dict=log_dict
            if log_dict is not None
            else table_steps.row
            if workdir and store_steps
            else None,
            writer=writer,
            **(sample_kwargs or {}),
        ):
            if energy == 'eq':
                steps.total = step + n_steps
                steps.set_description('evaluating')
                continue
            if energy is not None:
                steps.set_postfix(E=f'{energy:S}')
            if workdir:
                if len(blocks) > len(table_blocks['energy']):
                    block = blocks[-1]
                    table_blocks.row['energy'] = np.stack(
                        [unp.nominal_values(block), unp.std_devs(block)], -1
                    )
                h5file.flush()
            if step >= (steps.total or n_steps) - 1:
                break
    finally:
        steps.close()
        if workdir:
            writer.close()
            h5file.close()
    return {'energy': energy}
