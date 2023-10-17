from functools import partial

import jax
import jax.numpy as jnp
import kfac_jax

from deepqmc.clip import median_log_squeeze_and_mask
from deepqmc.optimizer import NoOptimizer

from .parallel import (
    gather_electrons_on_one_device,
    pexp_normalize_mean,
    pmap,
    pmean,
    rng_iterator,
    select_one_device,
    split_on_devices,
)
from .types import TrainState
from .utils import masked_mean

__all__ = ()


def fit_wf(  # noqa: C901
    rng,
    hamil,
    ansatz,
    opt,
    molecule_idx_sampler,
    sampler,
    electron_batch_size,
    steps,
    train_state,
    *,
    clip_mask_fn=None,
):
    device_count = jax.device_count()

    @partial(jax.custom_jvp, nondiff_argnums=(1, 2))
    def loss_fn(params, rng, batch):
        phys_conf, weight = batch
        rng = jax.random.split(rng, len(weight))
        rng_batch = jax.vmap(partial(jax.random.split, num=weight.shape[1]))(rng)
        E_loc, hamil_stats = jax.vmap(
            jax.vmap(hamil.local_energy(partial(ansatz.apply, params)))
        )(rng_batch, phys_conf)
        loss = pmean(jnp.nanmean(E_loc * weight))
        stats = {
            'E_loc/mean': jnp.nanmean(E_loc, axis=1),
            'E_loc/std': jnp.nanstd(E_loc, axis=1),
            'E_loc/min': jnp.nanmin(E_loc, axis=1),
            'E_loc/max': jnp.nanmax(E_loc, axis=1),
            **{
                k_hamil: v_hamil.mean(axis=1)
                for k_hamil, v_hamil in hamil_stats.items()
            },
        }
        return loss, (E_loc, stats)

    @loss_fn.defjvp
    def loss_jvp(rng, batch, primals, tangents):
        phys_conf, weight = batch
        loss, aux = loss_fn(*primals, rng, batch)
        E_loc, _ = aux
        E_loc_s, gradient_mask = jax.vmap(clip_mask_fn or median_log_squeeze_and_mask)(
            E_loc
        )
        E_mean = pmean((E_loc_s * weight).mean(axis=1))
        assert E_loc_s.shape == E_loc.shape, (
            f'Error with clipping function: shape of E_loc {E_loc.shape} '
            f'must equal shape of clipped E_loc {E_loc_s.shape}.'
        )
        assert gradient_mask.shape == E_loc.shape, (
            f'Error with masking function: shape of E_loc {E_loc.shape} '
            f'must equal shape of mask {gradient_mask.shape}.'
        )

        def log_likelihood(params):  # log(psi(theta))
            flat_phys_conf = jax.tree_util.tree_map(
                lambda x: x.reshape(-1, *x.shape[2:]), phys_conf
            )
            return jax.vmap(ansatz.apply, (None, 0))(params, flat_phys_conf).log

        log_psi, log_psi_tangent = jax.jvp(log_likelihood, primals, tangents)
        kfac_jax.register_normal_predictive_distribution(log_psi[:, None])
        loss_tangent = (
            (E_loc_s - E_mean[:, None]) * log_psi_tangent.reshape(E_loc.shape) * weight
        )
        loss_tangent = masked_mean(loss_tangent, gradient_mask)
        return (loss, aux), (loss_tangent, aux)
        # jax.custom_jvp has actually no official support for auxiliary output.
        # the second aux in the tangent output should be in fact aux_tangent.
        # we just output the same thing to satisfy jax's API requirement with
        # the understanding that we'll never need aux_tangent

    opt = opt(loss_fn)

    @pmap
    def sample_wf(state, rng, params, idxs):
        return sampler.sample(rng, state, partial(ansatz.apply, params), idxs)

    @pmap
    def update_sampler(state, params):
        return sampler.update(state, partial(ansatz.apply, params))

    def train_step(rng, step, smpl_state, params, opt_state):
        rng_sample, rng_kfac = split_on_devices(rng, 2)
        mol_idxs = molecule_idx_sampler.sample()
        smpl_state, phys_conf, smpl_stats = sample_wf(
            smpl_state, rng_sample, params, mol_idxs
        )
        weight = pmap(pexp_normalize_mean)(
            smpl_state['log_weight'][jnp.arange(device_count)[:, None], mol_idxs]
            if 'log_weight' in smpl_state.keys()
            else jnp.zeros(
                (
                    device_count,
                    molecule_idx_sampler.batch_size,
                    electron_batch_size // device_count,
                )
            )
        )
        params, opt_state, E_loc, stats = opt.step(
            rng_kfac,
            params,
            opt_state,
            (phys_conf, weight),
        )
        if not isinstance(opt, NoOptimizer):
            # WF was changed in _step, update psi values stored in smpl_state
            smpl_state = update_sampler(smpl_state, params)
        stats = {**stats, **smpl_stats}
        return smpl_state, params, opt_state, E_loc, mol_idxs, stats

    smpl_state, params, opt_state = train_state
    if opt_state is None:
        rng, rng_sample, rng_opt = split_on_devices(rng, 3)
        idxs = molecule_idx_sampler.sample()
        _, init_phys_conf, _ = sample_wf(smpl_state, rng_sample, params, idxs)
        opt_state = opt.init(
            rng_opt,
            params,
            (
                init_phys_conf,
                jnp.ones(
                    (
                        device_count,
                        molecule_idx_sampler.batch_size,
                        electron_batch_size // device_count,
                    )
                ),
            ),
        )
    train_state = smpl_state, params, opt_state

    for step, rng in zip(steps, rng_iterator(rng)):
        *train_state, E_loc, mol_idxs, stats = train_step(rng, step, *train_state)
        yield step, TrainState(*train_state), gather_electrons_on_one_device(
            E_loc
        ), select_one_device(mol_idxs), stats
