from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import kfac_jax
import optax

from .kfacext import make_graph_patterns
from .parallel import (
    PMAP_AXIS_NAME,
    all_device_mean,
    all_device_median,
    all_device_quantile,
    gather_electrons_on_one_device,
    pexp_normalize_mean,
    pmap,
    pmean,
    rng_iterator,
    select_one_device,
    split_on_devices,
)
from .utils import log_squeeze, masked_mean, tree_norm
from .wf.base import init_wf_params

__all__ = ()

TrainState = namedtuple('TrainState', 'sampler params opt')


def median_clip_and_mask(x, clip_width, median_center, exclude_width=jnp.inf):
    clip_center = all_device_median(x) if median_center else all_device_mean(x)
    abs_diff = jnp.abs(x - clip_center)
    mad = all_device_mean(abs_diff)
    x_clip = jnp.clip(x, clip_center - clip_width * mad, clip_center + clip_width * mad)
    gradient_mask = abs_diff < exclude_width
    return x_clip, gradient_mask


def median_log_squeeze_and_mask(
    x, clip_width=1.0, quantile=0.95, exclude_width=jnp.inf
):
    x_median = all_device_median(x)
    x_diff = x - x_median
    x_abs_diff = jnp.abs(x_diff)
    quantile = all_device_quantile(x_abs_diff, quantile)
    width = clip_width * quantile
    x_clip = x_median + 2 * width * log_squeeze(x_diff / (2 * width))
    gradient_mask = x_abs_diff / quantile < exclude_width
    return x_clip, gradient_mask


def init_fit(rng, hamil, ansatz, sampler, electron_batch_size):
    params = init_wf_params(rng, hamil, ansatz)
    smpl_state = sampler.init(rng, partial(ansatz.apply, params), electron_batch_size)
    return params, smpl_state


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

    energy_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    if opt is None:

        @pmap
        def _step(_rng_opt, params, _opt_state, batch):
            loss, (E_loc, stats) = loss_fn(params, _rng_opt, batch)

            return params, None, E_loc, stats

    elif isinstance(opt, optax.GradientTransformation):

        @pmap
        def _step(rng, params, opt_state, batch):
            (loss, (E_loc, stats)), grads = energy_and_grad_fn(params, rng, batch)
            grads = pmean(grads)
            updates, opt_state = opt.update(grads, opt_state, params)
            param_norm, update_norm, grad_norm = map(
                tree_norm, [params, updates, grads]
            )
            params = optax.apply_updates(params, updates)
            stats = {
                'opt/param_norm': param_norm,
                'opt/grad_norm': grad_norm,
                'opt/update_norm': update_norm,
                **stats,
            }
            return params, opt_state, E_loc, stats

        @pmap
        def init_opt(rng, params, batch):
            opt_state = opt.init(params)
            return opt_state

    else:

        def _step(rng, params, opt_state, batch):
            params, opt_state, opt_stats = opt.step(
                params,
                opt_state,
                rng,
                batch=batch,
                momentum=0,
            )
            stats = {
                'opt/param_norm': opt_stats['param_norm'],
                'opt/grad_norm': opt_stats['precon_grad_norm'],
                'opt/update_norm': opt_stats['update_norm'],
                **opt_stats['aux'][1],
            }
            return (
                params,
                opt_state,
                opt_stats['aux'][0],
                stats,
            )

        def init_opt(rng, params, batch):
            opt_state = opt.init(
                params,
                rng,
                batch,
            )
            return opt_state

        def kfac_batch_size_extractor(batch):
            _, weights = batch
            return weights.shape[0] * weights.shape[1]

        opt = opt(
            value_and_grad_func=energy_and_grad_fn,
            l2_reg=0.0,
            value_func_has_aux=True,
            value_func_has_rng=True,
            auto_register_kwargs={'graph_patterns': make_graph_patterns()},
            include_norms_in_stats=True,
            estimation_mode='fisher_exact',
            num_burnin_steps=0,
            min_damping=1e-4,
            inverse_update_period=1,
            multi_device=True,
            pmap_axis_name=PMAP_AXIS_NAME,
            batch_size_extractor=kfac_batch_size_extractor,
        )

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
        params, opt_state, E_loc, stats = _step(
            rng_kfac,
            params,
            opt_state,
            (phys_conf, weight),
        )
        if opt is not None:
            # WF was changed in _step, update psi values stored in smpl_state
            smpl_state = update_sampler(smpl_state, params)
        stats = {**stats, **smpl_stats}
        return smpl_state, params, opt_state, E_loc, mol_idxs, stats

    smpl_state, params, opt_state = train_state
    if opt is not None and opt_state is None:
        rng, rng_sample, rng_opt = split_on_devices(rng, 3)
        idxs = molecule_idx_sampler.sample()
        _, init_phys_conf, _ = sample_wf(smpl_state, rng_sample, params, idxs)
        opt_state = init_opt(
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
