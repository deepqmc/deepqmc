import logging
from typing import Optional, Union

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jax.experimental.multihost_utils import broadcast_one_to_all

from ..parallel import align_rng_key_across_devices, replicate_on_devices
from ..types import KeyArray, Params, PhysicalConfiguration, SamplerState, Stats
from ..utils import better_where
from .base import ElectronSampler, ElectronWarp, NucleiSampler

log = logging.getLogger(__name__)


class MoleculeIdxSampler:
    r"""Sample molecule indexes for transferable training."""

    def __init__(
        self,
        rng: KeyArray,
        n_mols: int,
        batch_size: int,
        shuffle: Union[bool, str] = False,
    ):
        assert shuffle in [False, 'once', 'always']
        self.rng = broadcast_one_to_all(rng)
        self.n_mols = n_mols
        self.batch_size = batch_size
        self.state = 0
        self.shuffle = shuffle
        self.permutation = self.new_permutation()

    def sample(self) -> jax.Array:
        idx = jnp.array(
            range(self.state, min(self.state + self.batch_size, self.n_mols))
        )
        value = [self.permutation[idx]]
        if len(idx) < self.batch_size:
            self.permutation = self.new_permutation()
            idx = jnp.array(range(self.batch_size - len(idx)))
            value.append(self.permutation[idx])
        self.state = (self.state + self.batch_size) % self.n_mols
        value = jnp.concatenate(value)
        return replicate_on_devices(value)

    def new_permutation(self) -> jax.Array:
        permutation = jnp.arange(self.n_mols)
        if self.shuffle:
            rng_next, rng = jax.random.split(self.rng)
            permutation = jax.random.permutation(rng, permutation)
            if self.shuffle == 'always':
                self.rng = rng_next
        return permutation


class MultiElectronicStateSampler:
    r"""Sample from multiple electronic states in parallel.

    This sampler applies ``vmap`` to an underlying
    :class:`~deepqmc.sampling.base.ElectronSampler` to sample from multiple electronic
    states in parallel.

    Args:
        sampler (~deepqmc.sampling.base.ElectronSampler): the electron sampler to use.
        n_state (int): the number of electronic states to sample from.
    """

    def __init__(self, sampler: ElectronSampler, n_state: int):
        self.sampler = sampler
        self.n_state = n_state

    def init(
        self, rng: KeyArray, params: Params, electron_batch_size: int, R: jax.Array
    ) -> SamplerState:
        rngs = jax.random.split(rng, self.n_state)
        smpl_state = jax.vmap(self.sampler.init, (0, 0, None, None))(
            rngs, params, electron_batch_size, R
        )
        return smpl_state

    def sample(
        self, rng: KeyArray, state: SamplerState, params: Params, R: jax.Array
    ) -> tuple[SamplerState, PhysicalConfiguration, Stats]:
        rngs = jax.random.split(rng, self.n_state)
        return jax.vmap(self.sampler.sample, (0, 0, 0, None))(rngs, state, params, R)

    def update(self, state: SamplerState, params: Params, R: jax.Array) -> SamplerState:
        return jax.vmap(self.sampler.update, (0, 0, None))(state, params, R)


class MultiNuclearGeometrySampler:
    r"""This sampler samples from multiple nuclear geometries in parallel.

    Args:
        elec_sampler (~deepqmc.sampling.electron_samplers.ElectronSampler):
            the electronic sampler to use
        nuc_sampler (~deepqmc.sampling.nuclei_samplers.NucleiSampler): the nuclei
            sampler to use.
        warp_elec_fn (~deepqmc.sampling.electron_samplers.ElectronWarp): the function
            that warps the electrons to the new nuclear geometry.
        update_nuc_period (int): optional, the number of steps between nuclear updates.
        elec_equilibration_steps (int): optional, the number of steps to equilibrate the
            electronic state after a nuclear update.
    """

    def __init__(
        self,
        elec_sampler: MultiElectronicStateSampler,
        nuc_sampler: NucleiSampler,
        warp_elec_fn: ElectronWarp,
        update_nuc_period: Optional[int],
        elec_equilibration_steps: Optional[int],
    ):
        self.elec_sampler = elec_sampler
        self.nuc_sampler = nuc_sampler
        self.warp_elec_fn = warp_elec_fn
        self.update_nuc_period = update_nuc_period
        self.elec_equilibration_steps = elec_equilibration_steps

    def init(
        self, rng: KeyArray, params: Params, electron_batch_size: int, R: jax.Array
    ):
        rngs = jax.random.split(rng, len(R))
        elec_smpl_state = jax.vmap(self.elec_sampler.init, (0, None, None, 0))(
            rngs, params, electron_batch_size, R
        )
        nuc_smpl_state = jax.vmap(self.nuc_sampler.init)(R)
        smpl_state = {
            'nuc': nuc_smpl_state,
            'elec': elec_smpl_state,
            'update_nuc_counter': jnp.zeros(len(R)),
        }
        return smpl_state

    def update_nuc(
        self, rng: KeyArray, smpl_state: SamplerState, params: Params
    ) -> tuple[SamplerState, Stats]:
        rng_nuc, rng_warp, rng_eq = jax.random.split(rng, 3)
        rng_nuc = align_rng_key_across_devices(rng_nuc)
        # required to maintain the paradigm of shared geometries across devices
        smpl_state['nuc'], dR, stats = self.nuc_sampler.sample(
            rng_nuc, smpl_state['nuc']
        )
        smpl_state['elec'] = self.warp_elec_fn(
            rng_warp, smpl_state['nuc']['R'], dR, smpl_state['elec']
        )
        smpl_state['elec'] = self.elec_sampler.update(
            smpl_state['elec'], params, smpl_state['nuc']['R']
        )
        if self.elec_equilibration_steps is not None:
            smpl_state['elec'] = jax.lax.fori_loop(
                0,
                self.elec_equilibration_steps,
                lambda i, state: self.elec_sampler.sample(
                    jax.random.fold_in(rng_eq, i),
                    state,
                    params,
                    smpl_state['nuc']['R'],
                )[0],
                smpl_state['elec'],
            )
        return smpl_state, stats

    def sample(
        self,
        rng: KeyArray,
        smpl_state: SamplerState,
        params: Params,
        mol_idxs: jax.Array,
    ) -> tuple[SamplerState, PhysicalConfiguration, Stats]:
        rngs_elec, rngs_nuc = jax.random.split(rng, (2, len(mol_idxs)))
        counter = smpl_state.pop('update_nuc_counter')
        smpl_state_it = jax.tree_util.tree_map(lambda x: x[mol_idxs], smpl_state)
        if self.update_nuc_period is not None:
            condition = counter[mol_idxs] == self.update_nuc_period - 1
            smpl_state_it = jax.lax.cond(
                jnp.any(condition),
                jax.vmap(lambda r, s: self.update_nuc(r, s, params)[0]),
                lambda r, s: s,
                rngs_nuc,
                smpl_state_it,
            )
            smpl_state_it = jax.tree_util.tree_map(
                lambda a, b: better_where(condition, a, b[mol_idxs]),
                smpl_state_it,
                smpl_state,
            )
            smpl_state = jax.tree_util.tree_map(
                lambda x, y: x.at[mol_idxs].set(y), smpl_state, smpl_state_it
            )
            counter = counter.at[mol_idxs].set(
                jnp.where(condition, 0, counter[mol_idxs] + 1)
            )
        smpl_state_it['elec'], phys_conf, stats = jax.vmap(
            self.elec_sampler.sample, (0, 0, None, 0)
        )(rngs_elec, smpl_state_it['elec'], params, smpl_state_it['nuc']['R'])
        smpl_state = jax.tree_util.tree_map(
            lambda x, y: x.at[mol_idxs].set(y), smpl_state, smpl_state_it
        )
        smpl_state['update_nuc_counter'] = counter
        batch_mol_idxs = jnp.tile(
            jnp.expand_dims(mol_idxs, range(1, len(phys_conf.batch_shape))),
            (1, *phys_conf.batch_shape[1:]),
        )
        phys_conf = jdc.replace(phys_conf, mol_idx=batch_mol_idxs)
        return smpl_state, phys_conf, stats

    def update(self, smpl_state: SamplerState, params: Params) -> SamplerState:
        smpl_state['elec'] = jax.vmap(self.elec_sampler.update, (0, None, 0))(
            smpl_state['elec'], params, smpl_state['nuc']['R']
        )
        return smpl_state
