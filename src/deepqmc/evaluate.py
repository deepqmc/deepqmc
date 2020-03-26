from itertools import count
from pathlib import Path

import tables
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .sampling import LangevinSampler, sample_wf

__version__ = '0.1.0'
__all__ = ['evaluate']


def evaluate(
    wf,
    store_coords=False,
    workdir=None,
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
        n_steps (int): number of sampling steps
        sample_size (int): number of Markov-chain walkers
        n_decorrelate (int): number of extra steps between samples included
            in the expectation value averaging
        sampler_kwargs (dict): extra arguments passed to
            :class:`~deepqmc.sampling.LangevinSampler`
        sample_kwargs (dict): extra arguments passed to
            :func:`~deepqmc.sampling.sample_wf`
        workdir (str): path where to store Tensorboard event file and HDF5 file with
            sampling block energies
        store_coords (bool): whether to store sampled electron coordinates

    Returns:
        dict: Expectation values with standard errors.
    """
    if workdir:
        workdir = Path(workdir)
        writer = SummaryWriter(log_dir=workdir, flush_secs=15)
        h5file = tables.open_file(workdir / 'blocks.h5', 'a')
        if 'blocks' not in h5file.root:
            table_blocks = h5file.create_table(
                '/', 'blocks', {'energy': tables.Float32Col((sample_size, 2))}
            )
            table_steps = h5file.create_table(
                '/',
                'steps',
                {'coords': tables.Float32Col((sample_size, wf.n_up + wf.n_down, 3))},
            )
        else:
            table_blocks = h5file.root['blocks']
            table_steps = h5file.root['steps']
    else:
        writer = None
    sampler = LangevinSampler.from_mf(
        wf,
        sample_size=sample_size,
        writer=writer,
        n_discard=0,
        **{'n_decorrelate': 4, **(sampler_kwargs or {})},
    )
    steps = tqdm(count(), desc='equilibrating')
    try:
        for step, energy, rs in sample_wf(
            wf,
            sampler.iter_with_info(),
            steps,
            log_dict=table_blocks.row if workdir else None,
            writer=writer,
            **(sample_kwargs or {}),
        ):
            if energy is None:
                steps.total = step + n_steps
                steps.set_description('evaluating')
                continue
            steps.set_postfix(E=f'{energy:S}')
            if workdir:
                table_blocks.row.append()
                table_blocks.flush()
                if store_coords:
                    for rs in rs:
                        table_steps.row['coords'] = rs.cpu().numpy()
                        table_steps.row.append()
                    table_steps.flush()
            if step >= (steps.total or n_steps) - 1:
                break
    finally:
        steps.close()
        if workdir:
            writer.close()
            h5file.close()
    return {'energy': energy}
