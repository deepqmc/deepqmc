from functools import partial

import jax
import kfac_jax
import optax

from deepqmc.kfacext import batch_size_extractor, make_graph_patterns
from deepqmc.parallel import PMAP_AXIS_NAME, pmap, pmean
from deepqmc.utils import ConstantSchedule, InverseSchedule, tree_norm

__all__ = ()

DEFAULT_OPT_KWARGS = {
    'adam': {'learning_rate': 1.0e-3, 'b1': 0.9, 'b2': 0.9},
    'adamw': {'learning_rate': 1.0e-3, 'b1': 0.9, 'b2': 0.9},
    'kfac': {
        'learning_rate_schedule': InverseSchedule(0.05, 10000),
        'damping_schedule': ConstantSchedule(0.001),
        'norm_constraint': 0.001,
        'estimation_mode': 'fisher_exact',
        'num_burnin_steps': 0,
        'inverse_update_period': 1,
    },
}


class Optimizer:
    r"""Base class for deepqmc's optimizer wrapper classes."""

    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def init(self, rng, params, batch):
        r"""Initialize the optimizer state."""
        return None

    def step(self, rng, params, opt_state, batch):
        r"""Perform an optimization step."""
        raise NotImplementedError


class NoOptimizer(Optimizer):
    @partial(pmap, static_broadcasted_argnums=(0,))
    def step(self, rng, params, opt_state, batch):
        loss, (E_loc, stats) = self.loss_fn(params, rng, batch)

        return params, opt_state, E_loc, stats


class OptaxOptimizer(Optimizer):
    def __init__(self, loss_fn, *, optax_opt):
        self.energy_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        self.optax_opt = optax_opt

    @partial(pmap, static_broadcasted_argnums=(0,))
    def init(self, rng, params, batch):
        opt_state = self.optax_opt.init(params)
        return opt_state

    @partial(pmap, static_broadcasted_argnums=(0,))
    def step(self, rng, params, opt_state, batch):
        (loss, (E_loc, stats)), grads = self.energy_and_grad_fn(params, rng, batch)
        grads = pmean(grads)
        updates, opt_state = self.optax_opt.update(grads, opt_state, params)
        param_norm, update_norm, grad_norm = map(tree_norm, [params, updates, grads])
        params = optax.apply_updates(params, updates)
        stats = {
            'opt/param_norm': param_norm,
            'opt/grad_norm': grad_norm,
            'opt/update_norm': update_norm,
            **stats,
        }
        return params, opt_state, E_loc, stats


class KFACOptimizer(Optimizer):
    def __init__(self, loss_fn, *, kfac):
        self.kfac = kfac(
            value_and_grad_func=jax.value_and_grad(loss_fn, has_aux=True),
            l2_reg=0.0,
            value_func_has_aux=True,
            value_func_has_rng=True,
            auto_register_kwargs={'graph_patterns': make_graph_patterns()},
            include_norms_in_stats=True,
            multi_device=True,
            pmap_axis_name=PMAP_AXIS_NAME,
            batch_size_extractor=batch_size_extractor,
        )

    def init(self, rng, params, batch):
        opt_state = self.kfac.init(
            params,
            rng,
            batch,
        )
        return opt_state

    def step(self, rng, params, opt_state, batch):
        params, opt_state, opt_stats = self.kfac.step(
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
        return params, opt_state, opt_stats['aux'][0], stats


def wrap_optimizer(opt):
    if opt is None:
        return NoOptimizer
    elif isinstance(opt, optax.GradientTransformation):
        return partial(OptaxOptimizer, optax_opt=opt)
    else:
        return partial(KFACOptimizer, kfac=opt)


def optimizer_from_name(opt_name, opt_kwargs=None, wrap=True):
    opt_kwargs = DEFAULT_OPT_KWARGS.get(opt_name, {}) | (opt_kwargs or {})
    return (
        partial(kfac_jax.Optimizer, **opt_kwargs)
        if opt_name == 'kfac'
        else getattr(optax, opt_name)(**opt_kwargs)
    )


def construct_optimizer(opt, opt_kwargs, wrap=True):
    if isinstance(opt, str):
        opt = optimizer_from_name(opt, opt_kwargs)
        if wrap:
            opt = wrap_optimizer(opt)
    if wrap and opt is None:
        opt = wrap_optimizer(opt)
    return opt
