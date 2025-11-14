from functools import partial

import jax
import jax.numpy as jnp

from ..hamil import MolecularHamiltonian
from ..parallel import all_device_mean
from ..physics import evaluate_spin, make_stochastic_spin_raising_operator
from ..types import Ansatz, KeyArray, Params, PhysicalConfiguration, Stats, Weight
from ..utils import batched_vmap, masked_mean, weighted_std


def compute_spin_contributions(
    hamil: MolecularHamiltonian,
    ansatz: Ansatz,
    params: Params,
    phys_conf: PhysicalConfiguration,
    states: list[int] | None = None,
) -> jax.Array:
    r"""Compute a batch of spin contributions.

    Args:
        hamil (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the system.
        ansatz (~deepqmc.types.Ansatz): the Ansatz object.
        params (~deepqmc.types.Params): the current parameters of the Ansatz.
        phys_conf (~deepqmc.types.PhysicalConfiguration): a batch of input to the
            Ansatz.
        states: (list[int] | None): list of state indices to compute spin for. If None,
            compute spin for all states.

    Returns:
        jax.Array: the samplewise contributions to spin expectation value.
    """

    if states is None:
        states = list(range(phys_conf.batch_shape[1]))
    spin_contributions = []
    for state in states:
        state_params = jax.tree.map(lambda x: x[state], params)  # noqa: B023
        spin_contributions.append(
            jax.vmap(
                jax.vmap(evaluate_spin(hamil, ansatz.apply), (None, 0)),
                (None, 0),
            )(state_params, phys_conf[:, state])
        )
    return jnp.stack(spin_contributions, axis=1)


def compute_mean_spin(
    spin_contriutions: jax.Array,
    weight: Weight,
    states: list[int] | None = None,
) -> tuple[jax.Array, Stats]:
    r"""Compute the mean of a batch of spin contributions.

    Args:
        spin_contriutions (jax.Array): the batch of local spin_contributions.
        weight (~deepqmc.types.Weight): the weight of each sample in the batch.
        states: (list[int] | None): list of state indices to compute spin for. If None,
            compute spin for all states.

    Returns:
        tuple[jax.Array, ~deepqmc.types.Stats]: a tuple of spin expectation value
        and statistics.
    """
    if states is None:
        states = list(range(weight.shape[1]))
    state_weights = jnp.stack([weight[:, state] for state in states], axis=1)
    stats = {
        'spin/mean': jnp.average(spin_contriutions, axis=-1, weights=state_weights),
        'spin/std': weighted_std(spin_contriutions, axis=-1, weights=state_weights),
    }
    return all_device_mean(spin_contriutions * state_weights), stats


def compute_mean_spin_tangent(
    spin_contributions: jax.Array,
    weight: Weight,
    log_psi_tangent: jax.Array,
    gradient_mask: jax.Array,
    states: list[int] | None = None,
) -> jax.Array:
    r"""Compute the tangent of the spin with respect to the Ansatz parameters.

    Args:
        spin_contributions (jax.Array): a batch of spin contributions.
        weight (~deepqmc.types.Weight): the weights of each sample in the batch.
        log_psi_tangent (jax.Array): the jvp of the WF values with respect to the Ansatz
            parameters.
        gradient_mask (jax.Array): a boolean samplewise mask to apply to the gradients.
        states: (list[int] | None): list of state indices to compute spin for. If None,
            compute spin for all states.

    Returns:
        jax.Array: the jvp of the spin with respect to the Ansatz parameters.
    """
    if states is None:
        states = list(range(weight.shape[1]))
    state_weights = jnp.stack([weight[:, state] for state in states], axis=1)
    state_log_psi_tangent = jnp.stack(
        [log_psi_tangent[:, state] for state in states], axis=1
    )
    state_gradient_mask = jnp.stack(
        [gradient_mask[:, state] for state in states], axis=1
    )
    per_mol_state_mean_spin = all_device_mean(
        spin_contributions * state_weights, axis=-1, keepdims=True
    )
    spin_contributions_tangent = (
        (spin_contributions - per_mol_state_mean_spin)
        * state_log_psi_tangent
        * state_weights
    )
    mean_energy_tangent = masked_mean(spin_contributions_tangent, state_gradient_mask)
    return mean_energy_tangent


def compute_spin_raising_contributions(
    rng: KeyArray,
    hamil: MolecularHamiltonian,
    ansatz: Ansatz,
    phys_conf: PhysicalConfiguration,
    params: Params,
    batch_size: int | None = None,
    states: list[int] | None = None,
) -> jax.Array:
    r"""Compute a batch of spin raising operator contributions.

    Computes :math:`1 - \sum_{\alpha} \frac{\hat P_{\alpha\beta} \Psi}{\Psi}`
    where a single :math:`\beta` is sampled randomly from the spin down electrons.

    Args:
        rng (~deepqmc.types.KeyArray): a random key.
        hamil (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the system.
        ansatz (~deepqmc.types.Ansatz): the Ansatz object.
        params (~deepqmc.types.Params): the current parameters of the Ansatz.
        phys_conf (~deepqmc.types.PhysicalConfiguration): a batch of input to the
            Ansatz.
        batch_size (int or None): if specified, the batch size for the electron axis
            mapping. If None, use jax.vmap.
        states: (list[int] | None): list of state indices to compute spin for. If None,
            compute spin for all states.

    Returns:
        jax.Array: the samplewise contributions to the stochastic spin raising
            expectation value.
    """
    if states is None:
        states = list(range(phys_conf.batch_shape[1]))
    spin_batch_shape = (phys_conf.batch_shape[0], *phys_conf.batch_shape[2:])
    down_idx = jnp.broadcast_to(
        jax.random.randint(rng, (), hamil.n_up, hamil.n_up + hamil.n_down),
        spin_batch_shape,
    )
    electron_axis_mapper = (
        jax.vmap
        if batch_size is None
        else partial(batched_vmap, batch_size=batch_size // jax.device_count())
    )
    spin_raising_contributions = []
    for state in states:
        state_params = jax.tree.map(lambda x: x[state], params)  # noqa: B023
        spin_raising_contributions.append(
            jax.vmap(
                electron_axis_mapper(
                    make_stochastic_spin_raising_operator(hamil, ansatz.apply),
                    in_axes=(None, 0, 0),
                ),
                (None, 0, 0),
            )(state_params, phys_conf[:, state], down_idx)
        )
    return jnp.stack(spin_raising_contributions, axis=1)


def compute_mean_spin_raising_tangent(
    spin_raising_contributions: jax.Array,
    spin_raising_tangent: jax.Array,
    weight: Weight,
    log_psi_tangent: jax.Array,
    gradient_mask: jax.Array,
    states: list[int] | None = None,
) -> jax.Array:
    r"""Compute the tangent of the spin raising operator with respect to the parameters.

    Args:
        spin_raising_contributions (jax.Array): a batch of spin raising contributions.
        spin_raising_tangent (jax.Array): a batch of spin raising contribution tangents.
            This is the gradient of the local values of the spin raising contributions.
            Necessary, because the stochastic spin raising operator is not self-adjoint.
        weight (~deepqmc.types.Weight): the weights of each sample in the batch.
        log_psi_tangent (jax.Array): the jvp of the WF values with respect to the Ansatz
            parameters.
        gradient_mask (jax.Array): a boolean samplewise mask to apply to the gradients.
        states: (list[int] | None): list of state indices to compute spin for. If None,
            compute spin for all states.

    Returns:
        jax.Array: the jvp of the spin raising operator with respect to the
            Ansatz parameters.
    """
    if states is None:
        states = list(range(weight.shape[1]))
    state_weights = jnp.stack([weight[:, state] for state in states], axis=1)
    state_log_psi_tangent = jnp.stack(
        [log_psi_tangent[:, state] for state in states], axis=1
    )
    state_gradient_mask = jnp.stack(
        [gradient_mask[:, state] for state in states], axis=1
    )
    per_mol_state_mean_spin_raising = all_device_mean(
        spin_raising_contributions * state_weights, axis=-1, keepdims=True
    )
    self_adjoint_tangent = (
        spin_raising_contributions - per_mol_state_mean_spin_raising
    ) * state_log_psi_tangent
    total_tangent = (
        per_mol_state_mean_spin_raising
        * state_weights
        * (2 * self_adjoint_tangent + spin_raising_tangent)
    )
    mean_total_tangent = masked_mean(total_tangent, state_gradient_mask)
    return mean_total_tangent
