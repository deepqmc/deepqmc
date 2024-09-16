from functools import partial
from typing import Optional, Protocol, cast

import jax
import jax.numpy as jnp
import kfac_jax

from ..hamil import MolecularHamiltonian
from ..parallel import PMAP_AXIS_NAME
from ..types import (
    Ansatz,
    Batch,
    Energy,
    KeyArray,
    Params,
    PhysicalConfiguration,
    Stats,
)
from ..utils import tree_stack
from .clip import (
    LocalEnergyClipAndMaskFn,
    PsiRatioClipAndMaskFn,
    clip_local_energy,
    clip_psi_ratio,
)
from .energy import (
    compute_local_energy,
    compute_mean_energy,
    compute_mean_energy_tangent,
)
from .overlap import (
    OverlapGradientScaleFactory,
    compute_mean_overlap,
    compute_mean_overlap_tangent,
    compute_psi_ratio,
    no_scaling,
    scale_by_energy_gap,
    scale_by_energy_std,
    scale_by_max_gap_std,
)
from .spin import (
    compute_mean_spin,
    compute_mean_spin_tangent,
    compute_spin_contributions,
)

__all__ = ()


class LossFunction(Protocol):
    def __call__(
        self,
        params: list[Params],
        rng: KeyArray,
        batch: Batch,
    ) -> tuple[jax.Array, tuple[Energy, Optional[jax.Array], Stats]]: ...


class LossFunctionFactory(Protocol):
    def __call__(
        self,
        hamil: MolecularHamiltonian,
        ansatz: Ansatz,
    ) -> LossFunction: ...


class LossAndGradFunction(Protocol):
    def __call__(
        self,
        params: list[Params],
        rng: KeyArray,
        batch: Batch,
    ) -> tuple[
        tuple[jax.Array, tuple[Energy, Optional[jax.Array], Stats]],
        tuple[jax.Array, tuple[Energy, Optional[jax.Array], Stats]],
    ]: ...


def compute_log_psi_tangent(
    ansatz: Ansatz,
    phys_conf: PhysicalConfiguration,
    params: list[Params],
    params_tangent: list[Params],
) -> jax.Array:
    def flat_log_psi(flat_phys_conf, params):
        return jax.vmap(ansatz.apply, (None, 0))(params, flat_phys_conf).log

    n_batch_dims = len(phys_conf.batch_shape)
    log_psis, log_psi_tangents = [], []
    for i, (state_params, state_params_tangent) in enumerate(
        zip(params, params_tangent)
    ):
        flat_phys_conf = jax.tree_util.tree_map(
            partial(lambda i, x: x[:, i].reshape(-1, *x.shape[n_batch_dims:]), i),
            phys_conf,
        )
        log_psi, log_psi_tangent = jax.jvp(
            partial(flat_log_psi, flat_phys_conf),
            (state_params,),
            (state_params_tangent,),
        )
        log_psis.append(log_psi.reshape(phys_conf.batch_shape[::2]))
        log_psi_tangents.append(log_psi_tangent.reshape(phys_conf.batch_shape[::2]))

    log_psi = jnp.stack(log_psis, axis=1).reshape(-1)
    kfac_jax.register_normal_predictive_distribution(log_psi[:, None])
    log_psi_tangent = jnp.stack(log_psi_tangents, axis=1)
    return log_psi_tangent


def create_loss_fn(
    hamil: MolecularHamiltonian,
    ansatz: Ansatz,
    clip_mask_fn: LocalEnergyClipAndMaskFn,
    clip_mask_overlap_fn: Optional[PsiRatioClipAndMaskFn] = None,
    alpha: Optional[float] = None,
    spin_penalty: Optional[float] = None,
    scale_overlap_by: Optional[str] = None,
    sort_states_by: Optional[str] = None,
    min_gap_scale_factor: float = 0.1,
) -> LossFunction:
    overlap_scale_factory = {
        None: no_scaling,
        'energy_gap': cast(
            OverlapGradientScaleFactory,
            partial(scale_by_energy_gap, min_gap_scale_factor=min_gap_scale_factor),
        ),
        'energy_std': cast(
            OverlapGradientScaleFactory,
            partial(scale_by_energy_std, min_gap_scale_factor=min_gap_scale_factor),
        ),
        'max_gap_std': cast(
            OverlapGradientScaleFactory,
            partial(scale_by_max_gap_std, min_gap_scale_factor=min_gap_scale_factor),
        ),
    }[scale_overlap_by]
    sort_states_factory = {
        None: lambda x: jnp.broadcast_to(jnp.arange(x.shape[-1]), x.shape),
        'energy': lambda x: jnp.argsort(x, axis=-1),
    }[sort_states_by]

    @jax.custom_jvp
    def loss_fn(
        params: list[Params], rng: KeyArray, batch: Batch
    ) -> tuple[jax.Array, tuple[Energy, Optional[jax.Array], Stats]]:
        phys_conf, weight, data = batch
        stacked_params = tree_stack(params)
        local_energy, hamil_stats = compute_local_energy(
            rng, hamil, ansatz.apply, stacked_params, phys_conf
        )
        loss, energy_stats = compute_mean_energy(local_energy, weight)
        stats = hamil_stats | energy_stats
        if phys_conf.batch_shape[1] > 1:
            psi_ratio, psi_stats = compute_psi_ratio(ansatz, stacked_params, phys_conf)
            overlap_loss, overlap_stats = compute_mean_overlap(psi_ratio, weight)
            loss += alpha * overlap_loss
            stats |= psi_stats | overlap_stats
        else:
            psi_ratio = None
        if spin_penalty is not None:
            spin_contributions = compute_spin_contributions(
                hamil, ansatz, stacked_params, phys_conf
            )
            spin, spin_stats = compute_mean_spin(spin_contributions, weight)
            loss += spin_penalty * spin
            stats |= spin_stats
        local_energy = jax.lax.all_gather(local_energy, PMAP_AXIS_NAME)
        psi_ratio = jax.lax.all_gather(psi_ratio, PMAP_AXIS_NAME)
        return loss, (local_energy, psi_ratio, stats)

    @loss_fn.defjvp
    def loss_fn_jvp(
        primals: tuple[list[Params], KeyArray, Batch],
        tangents: tuple[list[Params], KeyArray, Batch],
    ) -> tuple[
        tuple[jax.Array, tuple[Energy, Optional[jax.Array], Stats]],
        tuple[jax.Array, tuple[Energy, Optional[jax.Array], Stats]],
    ]:
        params, rng, (phys_conf, weight, data) = primals
        params_tangent, *_ = tangents

        log_psi_tangent = compute_log_psi_tangent(
            ansatz, phys_conf, params, params_tangent
        )

        stacked_params = tree_stack(params)
        local_energy, hamil_stats = compute_local_energy(
            rng, hamil, ansatz.apply, stacked_params, phys_conf
        )
        loss, energy_stats = compute_mean_energy(local_energy, weight)
        stats = hamil_stats | energy_stats
        clipped_local_energy, gradient_mask = clip_local_energy(
            clip_mask_fn, local_energy
        )
        loss_tangent = compute_mean_energy_tangent(
            clipped_local_energy, weight, log_psi_tangent, gradient_mask
        )

        if phys_conf.batch_shape[1] > 1:
            assert clip_mask_overlap_fn is not None and alpha is not None
            assert data is not None
            data['ordering'] = sort_states_factory(data['energy_ewm'])
            psi_ratio, psi_stats = compute_psi_ratio(ansatz, stacked_params, phys_conf)
            overlap_loss, overlap_stats = compute_mean_overlap(psi_ratio, weight)
            stats |= psi_stats | overlap_stats
            clipped_psi_ratio, ratio_gradient_mask = clip_psi_ratio(
                clip_mask_overlap_fn, psi_ratio
            )
            _, clipped_overlap_stats = compute_mean_overlap(clipped_psi_ratio, weight)
            overlap_tangent = compute_mean_overlap_tangent(
                clipped_psi_ratio,
                weight,
                log_psi_tangent,
                ratio_gradient_mask,
                clipped_overlap_stats['overlap/pairwise/mean'],
                overlap_scale_factory,
                data,
            )
            loss += alpha * overlap_loss
            loss_tangent += alpha * overlap_tangent
        else:
            psi_ratio = None

        if spin_penalty is not None:
            spin_contributions = compute_spin_contributions(
                hamil, ansatz, stacked_params, phys_conf
            )
            spin, spin_stats = compute_mean_spin(spin_contributions, weight)
            stats |= spin_stats
            spin_tangent = compute_mean_spin_tangent(
                spin_contributions, weight, log_psi_tangent, gradient_mask
            )
            loss += spin_penalty * spin
            loss_tangent += spin_penalty * spin_tangent

        local_energy = jax.lax.all_gather(local_energy, PMAP_AXIS_NAME)
        psi_ratio = jax.lax.all_gather(psi_ratio, PMAP_AXIS_NAME)
        aux = (local_energy, psi_ratio, stats)
        return (loss, aux), (loss_tangent, aux)
        # jax.custom_jvp has actually no official support for auxiliary output.
        # the second aux in the tangent output should be in fact aux_tangent.
        # we just output the same thing to satisfy jax's API requirement with
        # the understanding that we'll never need aux_tangent

    return loss_fn
