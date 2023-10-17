from functools import partial

import jax
import kfac_jax

from deepqmc.clip import median_log_squeeze_and_mask
from deepqmc.parallel import all_device_mean
from deepqmc.utils import masked_mean

__all__ = ()


def compute_local_energy(rng, hamil, ansatz, params, phys_conf):
    rng = jax.random.split(rng, len(phys_conf))
    rng = jax.vmap(partial(jax.random.split, num=phys_conf.batch_shape[1]))(rng)
    local_energy, hamil_stats = jax.vmap(
        jax.vmap(hamil.local_energy(partial(ansatz.apply, params)))
    )(rng, phys_conf)
    stats = {
        'E_loc/mean': local_energy.mean(axis=1),
        'E_loc/std': local_energy.std(axis=1),
        'E_loc/min': local_energy.min(axis=1),
        'E_loc/max': local_energy.max(axis=1),
        **{k_hamil: v_hamil.mean(axis=1) for k_hamil, v_hamil in hamil_stats.items()},
    }
    return local_energy, stats


def clip_local_energy(clip_mask_fn, local_energy):
    return jax.vmap(clip_mask_fn)(local_energy)


def compute_log_psi_tangent(ansatz, phys_conf, params, params_tangent):
    flat_phys_conf = jax.tree_util.tree_map(
        lambda x: x.reshape(-1, *x.shape[2:]), phys_conf
    )

    def flat_log_psi(params):
        return jax.vmap(ansatz.apply, (None, 0))(params, flat_phys_conf).log

    log_psi, log_psi_tangent = jax.jvp(flat_log_psi, (params,), (params_tangent,))
    kfac_jax.register_normal_predictive_distribution(log_psi[:, None])
    return log_psi_tangent.reshape(phys_conf.mol_idx.shape)


def compute_mean_energy(local_energy, weight):
    return all_device_mean(local_energy * weight)


def compute_mean_energy_tangent(local_energy, weight, log_psi_tangent, gradient_mask):
    per_mol_mean_energy = all_device_mean(local_energy * weight, axis=-1, keepdims=True)
    local_energy_tangent = (
        (local_energy - per_mol_mean_energy) * log_psi_tangent * weight
    )
    mean_energy_tangent = masked_mean(local_energy_tangent, gradient_mask)
    return mean_energy_tangent


def create_energy_loss_fn(hamil, ansatz, clip_mask_fn):
    @jax.custom_jvp
    def loss_fn(params, rng, batch):
        phys_conf, weight = batch
        local_energy, stats = compute_local_energy(
            rng, hamil, ansatz, params, phys_conf
        )
        loss = compute_mean_energy(local_energy, weight)
        return loss, (local_energy, stats)

    @loss_fn.defjvp
    def loss_fn_jvp(primals, tangents):
        params, rng, (phys_conf, weight) = primals
        params_tangent, *_ = tangents

        local_energy, stats = compute_local_energy(
            rng, hamil, ansatz, params, phys_conf
        )

        loss = compute_mean_energy(local_energy, weight)

        log_psi_tangent = compute_log_psi_tangent(
            ansatz, phys_conf, params, params_tangent
        )
        clipped_local_energy, gradient_mask = clip_local_energy(
            clip_mask_fn or median_log_squeeze_and_mask, local_energy
        )
        loss_tangent = compute_mean_energy_tangent(
            clipped_local_energy, weight, log_psi_tangent, gradient_mask
        )

        aux = (local_energy, stats)
        return (loss, aux), (loss_tangent, aux)
        # jax.custom_jvp has actually no official support for auxiliary output.
        # the second aux in the tangent output should be in fact aux_tangent.
        # we just output the same thing to satisfy jax's API requirement with
        # the understanding that we'll never need aux_tangent

    return loss_fn
