from itertools import count
from pathlib import Path

import h5py
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from uncertainties import unumpy as unp

from .sampling import LangevinSampler, sample_wf

__version__ = '0.1.0'
__all__ = ['evaluate']


def evaluate(
    wf,
    *,
    n_steps=500,
    sample_size=1_000,
    n_decorrelate=4,
    sampler_kwargs=None,
    sample_kwargs=None,
    cwd=None,
    store_coords=False,
):
    r"""Evaluate a wave function model.

    This is a top-level API function that rigorously evaluates a trained wave
    function model. It initializes a :class:`~deepqmc.sampling.LangevinSampler`,
    sets up a Tensorboard writer, and calls :func:`~deepqmc.sampling.sample_wf`.

    Args:
        wf (:class:`~deepqmc.wf.WaveFunction`): wave function model to be evaluated
        n_steps (int): number of sampling steps
        sample_size (int): number of Markov-chain walkers
        n_decorrelate (int): number of extra steps between samples included
            in the expectation value averaging
        sampler_kwargs (dict): extra arguments passed to
            :class:`~deepqmc.sampling.LangevinSampler`
        sample_kwargs (dict): extra arguments passed to
            :func:`~deepqmc.sampling.sample_wf`
        cwd (str): path where to store Tensorboard event file and HDF5 file with
            sampling block energies
        store_coords (bool): whether to store sampled electron coordinates

    Returns:
        dict: Expectation values with standard errors.
    """
    writer = SummaryWriter(log_dir=cwd, flush_secs=15)
    if cwd:
        block_file = h5py.File(Path(cwd) / 'blocks.h5', 'a', libver='v110')
        if 'energy' not in block_file:
            block_file.create_group('energy')
            for label in ['value', 'error']:
                block_file['energy'].create_dataset(
                    label, (0, sample_size), maxshape=(None, sample_size)
                )
            if store_coords:
                block_file.create_dataset(
                    'coord',
                    (0, sample_size, wf.n_up + wf.n_down, 3),
                    maxshape=(None, sample_size, wf.n_up + wf.n_down, 3),
                )
        block_file.swmr_mode = True
    sampler = LangevinSampler.from_mf(
        wf,
        sample_size=sample_size,
        n_discard=0,
        n_decorrelate=n_decorrelate,
        writer=writer,
        **(sampler_kwargs or {}),
    )
    blocks = []
    steps = tqdm(count(), desc='equilibrating')
    try:
        for step, energy, rs in sample_wf(
            wf,
            sampler.iter_with_info(),
            steps,
            writer=writer,
            blocks=blocks,
            **(sample_kwargs or {}),
        ):
            if energy is None:
                steps.total = step + n_steps
                steps.set_description('evaluating')
                continue
            steps.set_postfix(E=f'{energy:S}')
            if cwd:
                for key, val in [
                    ('energy/value', unp.nominal_values(blocks[-1])),
                    ('energy/error', unp.std_devs(blocks[-1])),
                ]:
                    ds = block_file[key]
                    ds.resize(ds.shape[0] + 1, axis=0)
                    ds[-1, :] = val
                if store_coords:
                    ds = block_file['coord']
                    ds.resize(ds.shape[0] + len(rs), axis=0)
                    ds[-len(rs) :, ...] = rs.cpu()
                block_file.flush()
            if step >= (steps.total or n_steps) - 1:
                break
    finally:
        writer.close()
        steps.close()
        if cwd:
            block_file.close()
    return {'energy': energy}
