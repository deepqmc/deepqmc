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
    gather_on_one_device,
    pexp_normalize_mean,
    pmap,
    pmean,
    replicate_on_devices,
    rng_iterator,
    split_on_devices,
)
from .utils import (
    log_squeeze,
    masked_mean,
    per_mol_stats,
    segment_nanmean,
    tree_norm,
)
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


def init_fit(rng, hamil, ansatz, sampler, sample_size):
    params = init_wf_params(rng, hamil, ansatz)
    smpl_state = sampler.init(rng, partial(ansatz.apply, params), sample_size)
    return params, smpl_state


def fit_wf(  # noqa: C901
    rng,
    hamil,
    ansatz,
    opt,
    sampler,
    sample_size,
    steps,
    train_state,
    *,
    clip_mask_fn=None,
    clip_mask_kwargs=None,
):
    stats_fn = partial(per_mol_stats, len(sampler))
    device_count = jax.device_count()

    @partial(jax.custom_jvp, nondiff_argnums=(1, 2))
    def loss_fn(params, rng, batch):
        phys_conf, weight = batch
        rng_batch = jax.random.split(rng, len(weight))
        E_loc, hamil_stats = jax.vmap(
            hamil.local_energy(partial(ansatz.apply, params))
        )(rng_batch, phys_conf)
        loss = pmean(jnp.nanmean(E_loc * weight))
        stats = {
            **stats_fn(E_loc, phys_conf.mol_idx, 'E_loc'),
            **{
                k_hamil: stats_fn(v_hamil, phys_conf.mol_idx, k_hamil, mean_only=True)
                for k_hamil, v_hamil in hamil_stats.items()
            },
        }
        return loss, (E_loc, stats)

    @loss_fn.defjvp
    def loss_jvp(rng, batch, primals, tangents):
        phys_conf, weight = batch
        loss, aux = loss_fn(*primals, rng, batch)
        E_loc, _ = aux
        E_loc_s, gradient_mask = (clip_mask_fn or median_log_squeeze_and_mask)(
            E_loc, **(clip_mask_kwargs or {})
        )
        E_mean = pmean(
            segment_nanmean(E_loc_s * weight, phys_conf.mol_idx, len(sampler)),
        )[phys_conf.mol_idx]
        assert E_loc_s.shape == E_loc.shape, (
            f'Error with clipping function: shape of E_loc {E_loc.shape} '
            f'must equal shape of clipped E_loc {E_loc_s.shape}.'
        )
        assert gradient_mask.shape == E_loc.shape, (
            f'Error with masking function: shape of E_loc {E_loc.shape} '
            f'must equal shape of mask {gradient_mask.shape}.'
        )

        def log_likelihood(params):  # log(psi(theta))
            return jax.vmap(ansatz.apply, (None, 0))(params, phys_conf).log

        log_psi, log_psi_tangent = jax.jvp(log_likelihood, primals, tangents)
        kfac_jax.register_normal_predictive_distribution(log_psi[:, None])
        loss_tangent = (E_loc_s - E_mean) * log_psi_tangent * weight
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

            return params, None, E_loc, {'per_mol': stats}

    elif isinstance(opt, optax.GradientTransformation):

        @pmap
        def _step(rng, params, opt_state, batch):
            (loss, (E_loc, per_mol_stats)), grads = energy_and_grad_fn(
                params, rng, batch
            )
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
                'per_mol': per_mol_stats,
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
                'per_mol': opt_stats['aux'][1],
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
        )

    @pmap
    def sample_wf(state, rng, params, select_idxs):
        return sampler.sample(rng, state, partial(ansatz.apply, params), select_idxs)

    @pmap
    def update_sampler(state, params):
        return sampler.update(state, partial(ansatz.apply, params))

    def train_step(rng, step, smpl_state, params, opt_state):
        rng_sample, rng_kfac = split_on_devices(rng)
        select_idxs = sampler.select_idxs(sample_size // device_count, step)
        select_idxs = replicate_on_devices(select_idxs)
        smpl_state, phys_conf, smpl_stats = sample_wf(
            smpl_state, rng_sample, params, select_idxs
        )
        weight = pmap(pexp_normalize_mean)(
            sampler.get_state(
                'log_weight',
                smpl_state,
                select_idxs,
                default=jnp.zeros((device_count, sample_size // device_count)),
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
        stats['per_mol'] = {**stats['per_mol'], **smpl_stats['per_mol']}
        return smpl_state, params, opt_state, E_loc, stats

    smpl_state, params, opt_state = train_state
    if opt is not None and opt_state is None:
        rng, rng_opt = split_on_devices(rng)
        init_select_idxs = sampler.select_idxs(sample_size // device_count, 0)
        init_select_idxs = replicate_on_devices(init_select_idxs)
        init_phys_conf = pmap(sampler.phys_conf)(smpl_state, init_select_idxs)
        opt_state = init_opt(
            rng_opt,
            params,
            (
                init_phys_conf,
                jnp.ones((device_count, sample_size // device_count)),
            ),
        )
    train_state = smpl_state, params, opt_state

    for step, rng in zip(steps, rng_iterator(rng)):
        *train_state, E_loc, stats = train_step(rng, step, *train_state)
        E_loc = gather_on_one_device(E_loc, flatten_device_axis=True)
        yield step, TrainState(*train_state), E_loc, stats
