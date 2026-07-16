from collections.abc import Callable, Iterable
from statistics import mean, stdev
from typing import Optional

import jax
import jax.numpy as jnp

from ..geom import pairwise_diffs
from ..hamil import MolecularHamiltonian
from ..molecule import Molecule
from ..parallel import pmap, rng_iterator, select_one_device
from ..types import (
    Ansatz,
    KeyArray,
    ParametrizedWaveFunction,
    Params,
    PhysicalConfiguration,
    SamplerState,
)
from .base import ElectronSampler
from .combined_samplers import (
    MoleculeIdxSampler,
    MultiElectronicStateSampler,
    MultiNuclearGeometrySampler,
)
from .nuclei_samplers import IdleNucleiSampler, no_elec_warp

__all__ = ['combine_samplers']


def chain(*samplers) -> ElectronSampler:
    r"""
    Combine multiple sampler types, to create advanced sampling schemes.

    For example :data:`chain(DecorrSampler(10),MetropolisSampler(hamil, tau=1.))`
    will create a :class:`MetropolisSampler`, where the samples
    are taken from every 10th MCMC step. The last element of the sampler chain has
    to be either a :class:`MetropolisSampler` or a :class:`LangevinSampler`.

    Args:
        samplers (~deepqmc.sampling.base.ElectronSampler): one or more sampler instances
            to combine.

    Returns:
        ~deepqmc.sampling.base.ElectronSampler: the combined sampler.
    """
    name = 'Sampler'
    bases = tuple(map(type, samplers))
    for base in bases:
        name = name.replace('Sampler', base.__name__)
    chained = type(name, bases, {'__init__': lambda self: None})()
    for sampler in samplers:
        chained.__dict__.update(sampler.__dict__)
    return chained  # type: ignore


def combine_samplers(
    samplers, hamil: MolecularHamiltonian, wf: ParametrizedWaveFunction
) -> ElectronSampler:
    r"""Combine samplers to create more advanced sampling schemes.

    Args:
        samplers (list[~deepqmc.sampling.base.ElectronSampler]): one or more sampler
            instances to combine.
        hamil (~deepqmc.hamil.MolecularHamiltonian): the molecular Hamiltonian.
        wf (~deepqmc.types.ParametrizedWaveFunction): the wave function to sample.
    """
    sampler = chain(*samplers[:-1], samplers[-1](hamil, wf))
    return sampler


def diffs_to_nearest_nuc(r, coords):
    z = pairwise_diffs(r, coords)
    idx = jnp.argmin(z[..., -1], axis=-1)
    return z[jnp.arange(len(r)), idx], idx


def crossover_parameter(z, f, charge):
    z, z2 = z[..., :3], z[..., 3]
    eps = jnp.finfo(f.dtype).eps
    z_unit = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
    f_unit = f / jnp.clip(jnp.linalg.norm(f, axis=-1, keepdims=True), eps, None)
    Z2z2 = charge**2 * z2
    return (1 + jnp.sum(f_unit * z_unit, axis=-1)) / 2 + Z2z2 / (10 * (4 + Z2z2))


def clean_force(force, phys_conf, mol, *, tau):
    z, idx = jax.vmap(diffs_to_nearest_nuc)(phys_conf.r, phys_conf.R)
    a = crossover_parameter(z, force, mol.charges[idx])
    av2tau = a * jnp.sum(force**2, axis=-1) * tau
    # av2tau can be small or zero, so the following expression must handle that
    factor = 2 / (jnp.sqrt(1 + 2 * av2tau) + 1)
    force = factor[..., None] * force
    eps = jnp.finfo(phys_conf.r.dtype).eps
    norm_factor = jnp.minimum(
        1.0,
        jnp.sqrt(z[..., -1])
        / (tau * jnp.clip(jnp.linalg.norm(force, axis=-1), eps, None)),
    )
    force = force * norm_factor[..., None]
    return force


def equilibrate(
    rng: KeyArray,
    params: Params,
    molecule_idx_sampler: MoleculeIdxSampler,
    sampler: MultiNuclearGeometrySampler,
    state: SamplerState,
    criterion: Callable[[PhysicalConfiguration], jax.Array],
    steps: Iterable[int],
    *,
    block_size: int,
    n_blocks: int = 5,
    allow_early_stopping: bool = True,
):
    r"""Run MCMC sampling steps until the walkers have equilibrated.

    A generator that repeatedly samples the wave function and, if
    ``allow_early_stopping`` is set, stops once ``criterion`` has stabilized: once
    ``block_size * n_blocks`` steps have been taken, the first and last
    ``block_size``-sized blocks of ``criterion`` values are compared, and sampling
    stops as soon as the difference between their means is smaller than the smaller
    of their two standard deviations.

    Args:
        rng (~deepqmc.types.KeyArray): key used for PRNG.
        params (~deepqmc.types.Params): the wave function parameters.
        molecule_idx_sampler (~deepqmc.sampling.MoleculeIdxSampler): samples the
            indices of the molecules to consider in each step.
        sampler (~deepqmc.sampling.MultiNuclearGeometrySampler): the sampler to
            equilibrate.
        state (~deepqmc.types.SamplerState): the initial sampler state.
        criterion (~collections.abc.Callable): a function of the sampled
            :class:`~deepqmc.types.PhysicalConfiguration` returning a scalar used to
            assess equilibration.
        steps (~collections.abc.Iterable[int]): the step indices to (potentially) run.
        block_size (int): the number of steps in each of the two compared blocks.
        n_blocks (int): the number of blocks worth of samples to buffer before the
            equilibration criterion is first evaluated.
        allow_early_stopping (bool): if :data:`False`, run through all of ``steps``
            regardless of ``criterion``.

    Yields:
        tuple: the current step, the updated sampler state, the sampled molecule
        indices, and the sampling statistics of that step.
    """
    sample_wf = pmap(sampler.sample)

    buffer_size = block_size * n_blocks
    buffer: list[float] = []
    for step, rng in zip(steps, rng_iterator(rng)):
        mol_idxs = molecule_idx_sampler.sample()
        state, phys_conf, stats = sample_wf(rng, state, params, mol_idxs)
        yield step, state, select_one_device(mol_idxs), stats
        if allow_early_stopping:
            buffer = [*buffer[-buffer_size + 1 :], criterion(phys_conf).item()]
            if len(buffer) < buffer_size:
                continue
            b1, b2 = buffer[:block_size], buffer[-block_size:]
            if abs(mean(b1) - mean(b2)) < min(stdev(b1), stdev(b2)):
                break


def initialize_sampling(
    rng: KeyArray,
    hamil: MolecularHamiltonian,
    ansatz: Ansatz,
    mols: list[Molecule],
    electronic_states: int,
    molecule_batch_size: int,
    *,
    elec_sampler,
    nuc_sampler=None,
    elec_warp_fn: Optional[Callable] = None,
    update_nuc_period: Optional[int] = None,
    elec_equilibration_steps: Optional[int] = None,
) -> tuple[MoleculeIdxSampler, MultiNuclearGeometrySampler]:
    r"""Assemble the molecule-index and combined electron/nuclear samplers.

    This is the function typically passed (as a :data:`~deepqmc.types.SamplerFactory`,
    partially applied with the sampler-specific keyword arguments) as the
    ``sampler_factory`` argument of :func:`~deepqmc.train.train`.

    Args:
        rng (~deepqmc.types.KeyArray): key used for PRNG.
        hamil (~deepqmc.hamil.MolecularHamiltonian): the molecular Hamiltonian.
        ansatz (~deepqmc.types.Ansatz): the wave function ansatz.
        mols (list[~deepqmc.molecule.Molecule]): the molecules to sample from.
        electronic_states (int): the number of electronic states to sample.
        molecule_batch_size (int): the number of molecules to sample in each step.
        elec_sampler (~collections.abc.Callable): a partially applied
            :class:`~deepqmc.sampling.base.ElectronSampler`, missing only the
            ``hamil`` and ``wf`` arguments, e.g. as created by
            :func:`~deepqmc.sampling.combine_samplers`.
        nuc_sampler (~collections.abc.Callable): optional, a partially applied
            :class:`~deepqmc.sampling.base.NucleiSampler`, missing only the
            ``charges`` argument. Defaults to
            :class:`~deepqmc.sampling.nuclei_samplers.IdleNucleiSampler`, i.e. fixed
            nuclear geometries.
        elec_warp_fn (~collections.abc.Callable): optional, a
            :class:`~deepqmc.sampling.base.ElectronWarp`, used to move electrons along
            with the nuclei when the nuclear geometry is updated. Defaults to
            :func:`~deepqmc.sampling.nuclei_samplers.no_elec_warp`.
        update_nuc_period (int): optional, the number of steps between nuclear
            geometry updates.
        elec_equilibration_steps (int): optional, the number of electron sampling
            steps to take between two nuclear geometry updates.

    Returns:
        tuple[~deepqmc.sampling.MoleculeIdxSampler,
        ~deepqmc.sampling.MultiNuclearGeometrySampler]: the molecule-index sampler and
        the combined electron/nuclear sampler.
    """
    molecule_idx_sampler = MoleculeIdxSampler(
        rng, len(mols), molecule_batch_size, 'once'
    )
    elec_sampler = elec_sampler(hamil=hamil, wf=ansatz.apply)
    multi_state_elec_sampler = MultiElectronicStateSampler(
        elec_sampler, electronic_states
    )
    nuc_sampler = (IdleNucleiSampler if nuc_sampler is None else nuc_sampler)(
        hamil.mol.charges,
    )
    elec_warp_fn = no_elec_warp if elec_warp_fn is None else elec_warp_fn
    sampler = MultiNuclearGeometrySampler(
        multi_state_elec_sampler,
        nuc_sampler,
        elec_warp_fn,
        update_nuc_period,
        elec_equilibration_steps,
    )
    return molecule_idx_sampler, sampler


def initialize_sampler_state(rng, sampler, params, electron_batch_size, nuc_coords):
    r"""Initialize the sampler state, split across the available devices.

    Args:
        rng (~deepqmc.types.KeyArray): key used for PRNG.
        sampler (~deepqmc.sampling.MultiNuclearGeometrySampler): the sampler to
            initialize.
        params (~deepqmc.types.Params): the wave function parameters.
        electron_batch_size (int): the total number of electron walkers to create,
            split evenly across all devices.
        nuc_coords (~jax.Array): the initial nuclear coordinates of the sampled
            molecule(s).

    Returns:
        ~deepqmc.types.SamplerState: the initialized, device-sharded sampler state.
    """

    @jax.pmap
    def sampler_state_initializer(rng, params):
        return sampler.init(
            rng,
            params,
            electron_batch_size // jax.device_count(),
            nuc_coords,
        )

    return sampler_state_initializer(rng, params)
