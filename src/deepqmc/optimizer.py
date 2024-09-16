from functools import partial
from typing import Optional, Protocol, TypeVar, cast

import jax
import jax.numpy as jnp
import optax

from .kfacext import batch_size_extractor, make_graph_patterns
from .loss.loss_function import LossAndGradFunction
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
        merge_keys: Optional[list[str]] = None,
    ):
        r"""Initializes the optimizer object.

        Args:
            loss_and_grad_fn (~deepqmc.loss.loss_function.LossAndGradFunction):
                a function that returns the loss and the gradient with respect to
                the model parameters alongside auxiliary data.
            merge_keys (list[str]): a list of keys for wave function parameters that
                are merged across ansatzes for multiple electronic states.
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
            jax.Array | None, ~deepqmc.types.Stats]: the new model
            parameters, an updated optimizer state, the energies obtained during the
            evaluation of the loss function, if applicable the wave function ratios
            obtained during the evaluation of the loss dunction and further statistics.
        """
        ...


class NoOptimizer(Optimizer):
    def __init__(
        self,
        loss_and_grad_fn: LossAndGradFunction,
        merge_keys: Optional[list[str]] = None,
    ):
        self.loss_and_grad_fn = loss_and_grad_fn

    @partial(pmap, static_broadcasted_argnums=(0,))
    def step(
        self, rng: KeyArray, params: Params, opt_state: OptState, batch: Batch
    ) -> tuple[Params, OptState, Energy, Optional[jax.Array], Stats]:
        (loss, (E_loc, ratios, stats)), _ = self.loss_and_grad_fn(
            tree_unstack(params), rng, batch
        )

        return params, opt_state, E_loc, ratios, stats


class OptaxOptimizer(Optimizer):
    def __init__(
        self,
        loss_and_grad_fn: LossAndGradFunction,
        merge_keys: Optional[list[str]] = None,
        *,
        optax_opt,
    ):
        self.energy_and_grad_fn = loss_and_grad_fn
        self.merge_keys = merge_keys
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
        (loss, (E_loc, ratios, stats)), grads = self.energy_and_grad_fn(
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
        params = merge_states(tree_stack(params_list), self.merge_keys)
        stats = {
            'opt/param_norm': param_norm,
            'opt/grad_norm': grad_norm,
            'opt/update_norm': update_norm,
            **stats,
        }
        return params, opt_state, E_loc, ratios, stats


class KFACOptimizer(Optimizer):
    def __init__(
        self, loss_and_grad_fn, merge_keys: Optional[list[str]] = None, *, kfac
    ):
        self.kfac = kfac(
            value_and_grad_func=loss_and_grad_fn,
            l2_reg=0.0,
            value_func_has_aux=True,
            value_func_has_rng=True,
            auto_register_kwargs={'graph_patterns': make_graph_patterns()},
            include_norms_in_stats=True,
            multi_device=True,
            pmap_axis_name=PMAP_AXIS_NAME,
            batch_size_extractor=batch_size_extractor,
        )
        self.merge_keys = merge_keys

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
        params = self.pmap_merge_states(
            self.pmap_tree_stack(params_list), self.merge_keys
        )
        stats = {
            'opt/param_norm': opt_stats['param_norm'],
            'opt/grad_norm': opt_stats['precon_grad_norm'],
            'opt/update_norm': opt_stats['update_norm'],
            **opt_stats['aux'][2],
        }
        return params, opt_state, opt_stats['aux'][0], opt_stats['aux'][1], stats

    @partial(jax.pmap, static_broadcasted_argnums=(0,))
    def pmap_tree_stack(self, trees: list[T]) -> T:
        return tree_stack(trees)

    @partial(jax.pmap, static_broadcasted_argnums=(0,))
    def pmap_tree_unstack(self, tree: T) -> list[T]:
        return tree_unstack(tree)

    @partial(jax.pmap, static_broadcasted_argnums=(0, 2))
    def pmap_merge_states(
        self, params: Params, keys_whitelist: Optional[list[str]]
    ) -> Params:
        return merge_states(params, keys_whitelist)


def merge_states(params: Params, merge_keys: Optional[list[str]]) -> Params:
    """Averages the parameters along the state axis."""
    av = lambda x: jnp.mean(x, axis=0, keepdims=True).repeat(x.shape[0], axis=0)
    params_filtered = filter_dict(params, merge_keys)
    params_averaged = jax.tree_map(av, params_filtered)
    return params | params_averaged
