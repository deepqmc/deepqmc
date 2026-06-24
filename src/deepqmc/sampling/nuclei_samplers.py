from collections.abc import Callable, Iterable
from typing import Optional

import jax
import jax.numpy as jnp

from ..geom.coordinate_transform import (
    CartesianCoordinateTransform,
    InvertibleCoordinateTransform,
)
from ..geom.zmatrix import StochasticZMatrixTemplate
from ..physics import pairwise_distance
from ..types import KeyArray, SamplerState, Stats


class IdleNucleiSampler:
    r"""
    Keeps track of nuclei without updating positions.

    Args:
        nuc_coords (~jax.Array): initial coordinates of the sampled molecules


    """

    def __init__(self, charges: jax.Array):
        pass

    def init(self, nuc_coords: jax.Array, *args, **kwargs) -> SamplerState:
        state = {'R': nuc_coords}
        return state

    def sample(
        self, rng: KeyArray, state: SamplerState
    ) -> tuple[SamplerState, jax.Array, Stats]:
        return state, jnp.zeros_like(state['R']), {}


class ConstraintNucleiSampler:
    r"""
    Samples nuclear positions around a fixed geometry.

    Args:
        charges (~jax.Array): the nuclear charges of the molecule (:math:`N_\text{nuc}`).
        noise_fn (~collections.abc.Callable | list[~collections.abc.Callable]): a noise
            distribution (or per-coordinate list of distributions) to sample
            displacements from. Each callable must have the signature
            ``(rng, shape) -> ~jax.Array``. Defaults to :func:`jax.random.normal`.
        coordinate_transform
            (~deepqmc.geom.coordinate_transform.InvertibleCoordinateTransform):
            optional, an invertible coordinate transform applied before adding noise.
            Defaults to a plain Cartesian transform.
        constraints (list | None): optional, a list of constraints of the form
            ``(idxs_at, idxs_set, fn)``.
    """

    def __init__(
        self,
        charges: jax.Array,
        *,
        noise_fn: (
            Callable[[KeyArray, tuple], jax.Array]
            | list[Callable[[KeyArray, tuple], jax.Array]]
        ) = jax.random.normal,
        coordinate_transform: Optional[InvertibleCoordinateTransform] = None,
        constraints: Optional[list] = None,
    ):
        def constraint_fn(update):
            for idxs_at, idxs_set, fn in constraints or []:
                fn = fn if fn is not None else lambda x, y: y
                idxs_at = idxs_at if idxs_at is not None else slice(None)
                idxs_set = idxs_set if idxs_set is not None else idxs_at
                update = update.at[idxs_at].set(fn(update[idxs_at], update[idxs_set]))
            return update

        self.constraint_fn = constraint_fn
        self.noise_fn = noise_fn
        self.coordinate_transform = (
            CartesianCoordinateTransform(len(charges))
            if coordinate_transform is None
            else coordinate_transform
        )

    def init(self, nuc_coords: jax.Array, *args, **kwargs) -> SamplerState:
        state = {'R': nuc_coords, 'R0': nuc_coords}
        return state

    def sample(
        self, rng: KeyArray, state: SamplerState
    ) -> tuple[SamplerState, jax.Array, Stats]:
        noise = (
            jnp.concatenate(
                [
                    fn(jax.random.fold_in(rng, i), (1,))
                    for i, fn in enumerate(self.noise_fn)
                ]
            )
            if isinstance(self.noise_fn, Iterable)
            else self.noise_fn(rng, (len(self.coordinate_transform),))
        )
        update = self.constraint_fn(noise)
        internal_coords = self.coordinate_transform.from_cartesian(state['R0'])
        state['R'] = self.coordinate_transform.to_cartesian(internal_coords + update)
        # TODO: combine with deepqmc_pub_transferabiliry coordinate handlers?
        return state, state['R'] - state['R0'], {}


class PermutationNucleiSampler:
    r"""Nuclei sampler that permutes nuclei with the same atomic number."""

    def __init__(self, charges: jax.Array):
        charges = jnp.asarray(charges)
        self.n_nuc = len(charges)
        self.nuc_type_idxs = [
            jnp.arange(len(charges))[charges == nuc_type]
            for nuc_type in jnp.unique(charges)
        ]

    def permutation(self, rng: KeyArray) -> jax.Array:
        idx = jnp.arange(self.n_nuc)
        for nuc_type_idx in self.nuc_type_idxs:
            idx = idx.at[nuc_type_idx].set(jax.random.permutation(rng, nuc_type_idx))
        return idx

    def init(self, nuc_coords: jax.Array, *args, **kwargs) -> SamplerState:
        state = {'R': nuc_coords}
        return state

    def sample(
        self, rng: KeyArray, state: SamplerState
    ) -> tuple[SamplerState, jax.Array, Stats]:
        idx = self.permutation(rng)
        R_old = state['R']
        state['R'] = state['R'][idx]
        dR = state['R'] - R_old
        return state, dR, {}


class ZMatrixSampler:
    r"""Nuclei sampler sampling nuclei positions using a Z-matrix."""

    def __init__(
        self, charges: jax.Array, *, z_matrix_template: StochasticZMatrixTemplate
    ):
        self.z_matrix_template = z_matrix_template

    def init(self, nuc_coords: jax.Array, *args, **kwargs) -> SamplerState:
        state = {'R': nuc_coords, 'R0': nuc_coords}
        return state

    def sample(
        self, rng: KeyArray, state: SamplerState
    ) -> tuple[SamplerState, jax.Array, Stats]:
        R_old = state['R']
        state['R'] = self.z_matrix_template.concretize_from_cartesian(state['R0'])(
            rng
        ).to_cartesian()
        dR = state['R'] - R_old
        return state, dR, {}


def no_elec_warp(
    rng: KeyArray, R: jax.Array, dR: jax.Array, smpl_state: SamplerState
) -> SamplerState:
    r"""Identity electron warp function."""
    return smpl_state


def nn_elec_warp(
    rng: KeyArray, R: jax.Array, dR: jax.Array, smpl_state: SamplerState
) -> SamplerState:
    r"""Nearest neighbor electron warp function."""
    R_old = R - dR
    dists = pairwise_distance(R_old[..., None, None, :, :], smpl_state['r'])
    mn = jnp.argmin(dists, axis=-2)
    smpl_state['r'] += dR[mn]
    return smpl_state


def fn_elec_warp(
    rng: KeyArray,
    R: jax.Array,
    dR: jax.Array,
    smpl_state: SamplerState,
    fn: Callable[[jax.Array], jax.Array],
) -> SamplerState:
    r"""Electron warp function using a user-defined distance scaling function."""
    R_old = R - dR
    dists = pairwise_distance(R_old[..., None, None, :, :], smpl_state['r'])
    dR = jnp.einsum(
        'jk, mnjo -> mnok', dR, (fn(dists) / fn(dists).sum(-2, keepdims=True))
    )
    smpl_state['r'] += dR
    return smpl_state
