from functools import partial
from typing import Optional, Protocol, TypeVar, cast

import jax
import jax.numpy as jnp
import optax

from .kfacext import batch_size_extractor
from .loss import LossAndGradFunction
from .parallel import PMAP_AXIS_NAME, pmap, pmean
from .types import Batch, Energy, KeyArray, OptState, Params, Stats
from .utils import filter_dict, tree_norm, tree_stack, tree_unstack

__all__ = ['Optimizer']

T = TypeVar('T')


class Optimizer(Protocol):
    r"""Protocol for :class:`~deepqmc.optimizer.Optimizer` objects."""

    def __init__(
        self,
        loss_and_grad_fn: LossAndGradFunction,
    ):
        r"""Initializes the optimizer object.

        Args:
            loss_and_grad_fn (~deepqmc.loss.loss_function.LossAndGradFunction):
                a function that returns the loss and the gradient with respect to
                the model parameters alongside auxiliary data.
        """
        ...

    def init(self, rng: KeyArray, params: Params, batch: Batch) -> OptState:
        r"""Initialize the optimizer state.

        Args:
            rng (~deepqmc.types.KeyArray): the RNG key used to initialize random
                components the of optimizer state.
            params (~deepqmc.types.Params): the parameters of the wave function
                ansatz/ansatzes to be optimized during training.
            batch (~deepqmc.types.Batch): a tuple containing a physical configuration,
                a set of sample weights and auxiliary data.

        Returns:
            ~deepqmc.types.OptState: the initial state of the optimizer
        """
        ...

    def step(
        self, rng: KeyArray, params: Params, opt_state: OptState, batch: Batch
    ) -> tuple[Params, OptState, Energy, Optional[jax.Array], Stats]:
        r"""Perform an optimization step.

        Args:
            rng (~deepqmc.types.KeyArray): the RNG key for the optimizer update.
            params (~deepqmc.types.Params): the current parameters of the wave function
                ansatz/ansatzes.
            opt_state (~deepqmc.types.OptState): the current state of the optimizer
            batch (~deepqmc.types.Batch): a tuple containing a physical configuration,
                a set of sample weights and auxiliary data.

        Returns:
            tuple[~deepqmc.types.Params, ~deepqmc.types.OptState, ~deepqmc.types.Energy,
            ~jax.Array | None, ~deepqmc.types.Stats]: the new model
            parameters, an updated optimizer state, the energies obtained during the
            evaluation of the loss function, if applicable the wave function ratios
            obtained during the evaluation of the loss function and further statistics.
        """
        ...


class NoOptimizer(Optimizer):
    r"""Evaluation-only optimizer that freezes the wave function parameters.

    Implements the :class:`~deepqmc.optimizer.Optimizer` protocol without
    performing any parameter update.  The loss function is still evaluated on
    each step so that energies and wave function statistics are collected, but
    gradients are discarded and the parameters are returned unchanged.  Use
    this class to run inference with a trained ansatz.

    Args:
        loss_and_grad_fn (~deepqmc.loss.LossAndGradFunction): callable that
            returns the loss, local energies, and gradients.
    """

    def __init__(
        self,
        loss_and_grad_fn: LossAndGradFunction,
    ):
        self.loss_and_grad_fn = loss_and_grad_fn

    @partial(pmap, static_broadcasted_argnums=(0,))
    def step(
        self, rng: KeyArray, params: Params, opt_state: OptState, batch: Batch
    ) -> tuple[Params, OptState, Energy, Optional[jax.Array], Stats]:
        (_, (E_loc, ratios, stats)), _ = self.loss_and_grad_fn(
            tree_unstack(params), rng, batch
        )

        return params, opt_state, E_loc, ratios, stats


class OptaxOptimizer(Optimizer):
    r"""First-order optimizers bafrom the :mod:`optax` module.

    Wraps any :mod:`optax` optimizer and handles device-parallel gradient
    averaging (:func:`pmean`) and parameter stacking automatically.  Per-step
    statistics include ``opt/param_norm``, ``opt/grad_norm``, and
    ``opt/update_norm``.

    Args:
        loss_and_grad_fn (~deepqmc.loss.LossAndGradFunction): callable that
            returns the loss, local energies, and gradients.
        optax_opt: an :mod:`optax` optimizer instance (e.g.
            ``optax.adam(learning_rate=1e-3)``).
    """

    def __init__(
        self,
        loss_and_grad_fn: LossAndGradFunction,
        *,
        optax_opt,
    ):
        self.energy_and_grad_fn = loss_and_grad_fn
        self.optax_opt = optax_opt

    @partial(pmap, static_broadcasted_argnums=(0,))
    def init(self, rng: KeyArray, params: Params, batch: Batch) -> OptState:
        opt_state = self.optax_opt.init(tree_unstack(params))
        return opt_state

    @partial(pmap, static_broadcasted_argnums=(0,))
    def step(
        self, rng: KeyArray, params: Params, opt_state: OptState, batch: Batch
    ) -> tuple[Params, OptState, Energy, Optional[jax.Array], Stats]:
        params_list = tree_unstack(params)
        (_, (E_loc, ratios, stats)), grads = self.energy_and_grad_fn(
            params_list, rng, batch
        )
        grads = pmean(grads)
        updates, opt_state = self.optax_opt.update(grads, opt_state, params_list)
        param_norm, update_norm, grad_norm = map(
            tree_norm, [params_list, updates, grads]
        )
        params_list = optax.apply_updates(params_list, updates)
        params_list = cast(
            list[Params], params_list
        )  # optax.apply_updates overwrites our type
        params = tree_stack(params_list)
        stats = {
            'opt/param_norm': param_norm,
            'opt/grad_norm': grad_norm,
            'opt/update_norm': update_norm,
            **stats,
        }
        return params, opt_state, E_loc, ratios, stats


class KFACOptimizer(Optimizer):
    r"""Second-order optimizer using the KFAC method [Martens15]_.

    Wraps the :mod:`kfac_jax` optimizer and wires up the multi-device
    infrastructure required by DeepQMC (``pmap``, ``pmap_axis_name``, batch
    size extraction).

    Args:
        loss_and_grad_fn (~deepqmc.loss.LossAndGradFunction): callable that
            returns the loss, local energies, and gradients; passed directly to
            :mod:`kfac_jax` as ``value_and_grad_func``.
        kfac: a partially-initialized :mod:`kfac_jax` optimizer constructor,
            i.e. a callable that accepts ``value_and_grad_func`` and related
            keyword arguments and returns the optimizer object.
    """

    def __init__(self, loss_and_grad_fn, *, kfac):
        self.kfac = kfac(
            value_and_grad_func=loss_and_grad_fn,
            l2_reg=0.0,
            value_func_has_aux=True,
            value_func_has_rng=True,
            include_norms_in_stats=True,
            multi_device=True,
            pmap_axis_name=PMAP_AXIS_NAME,
            batch_size_extractor=batch_size_extractor,
        )

    def init(self, rng: KeyArray, params: Params, batch: Batch) -> OptState:
        opt_state = self.kfac.init(
            self.pmap_tree_unstack(params),
            rng,
            batch,
        )
        return opt_state

    def step(
        self, rng, params: Params, opt_state: OptState, batch: Batch
    ) -> tuple[Params, OptState, Energy, Optional[jax.Array], Stats]:
        params_list, opt_state, opt_stats = self.kfac.step(
            self.pmap_tree_unstack(params),
            opt_state,
            rng,
            batch=batch,
            momentum=0,
        )
        params = self.pmap_tree_stack(params_list)
        stats = {
            'opt/param_norm': opt_stats['param_norm'],
            'opt/grad_norm': opt_stats['precon_grad_norm'],
            'opt/update_norm': opt_stats['update_norm'],
            'opt/scaled_grad_norm_sq': opt_stats['scaled_grad_norm_sq'],
            **opt_stats['aux'][2],
        }
        return params, opt_state, opt_stats['aux'][0], opt_stats['aux'][1], stats

    @partial(jax.pmap, static_broadcasted_argnums=(0,))
    def pmap_tree_stack(self, trees: list[T]) -> T:
        return tree_stack(trees)

    @partial(jax.pmap, static_broadcasted_argnums=(0,))
    def pmap_tree_unstack(self, tree: T) -> list[T]:
        return tree_unstack(tree)


def merge_states(params: Params, merge_keys: Optional[tuple[str, ...]]) -> Params:
    r"""Average selected parameters across electronic states.

    For each parameter key that contains at least one of the substrings in
    ``merge_keys``, the parameter tensor is averaged along the state axis
    (axis 0) and the result is broadcast back so all states share the same
    values.  Parameters whose keys do not match are left unchanged.  This is
    used to enforce weight-sharing across electronic states during training.

    Args:
        params (~deepqmc.types.Params): parameter pytree; the outermost axis of
            each leaf is the electronic-state axis.
        merge_keys (Optional[tuple[str, ...]]): substrings used to select which
            parameter groups to merge; if ``None`` no parameters are merged.

    Returns:
        ~deepqmc.types.Params: parameter pytree with the selected leaves
            replaced by their state-averaged values.
    """
    av = lambda x: jnp.mean(x, axis=0, keepdims=True).repeat(x.shape[0], axis=0)
    params_filtered = filter_dict(params, merge_keys)
    params_averaged = jax.tree.map(av, params_filtered)
    return params | params_averaged


@partial(jax.pmap, static_broadcasted_argnums=(1,))
def pmap_merge_states(params: Params, merge_keys: Optional[tuple[str, ...]]) -> Params:
    return merge_states(params, merge_keys)
