import jax
import jax.numpy as jnp

from ..hamil import MolecularHamiltonian
from ..parallel import all_device_mean
from ..physics import evaluate_spin, make_stochastic_spin_raising_operator
from ..types import Ansatz, KeyArray, Params, PhysicalConfiguration, Stats, Weight
from ..utils import masked_mean, weighted_std


def compute_spin_contributions(
    hamil: MolecularHamiltonian,
    ansatz: Ansatz,
    params: Params,
    phys_conf: PhysicalConfiguration,
) -> jax.Array:
    r"""Compute a batch of spin contributions.

    Args:
        hamil (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the system.
        ansatz (~deepqmc.types.Ansatz): the Ansatz object.
        params (~deepqmc.types.Params): the current parameters of the Ansatz.
        phys_conf (~deepqmc.types.PhysicalConfiguration): a batch of input to the
            Ansatz.

    Returns:
        jax.Array: the samplewise contributions to spin expectation value.
    """

    spin_contributions = jax.vmap(
        jax.vmap(jax.vmap(evaluate_spin(hamil, ansatz.apply), (None, 0))),
        (None, 0),
    )(params, phys_conf)
    return spin_contributions


def compute_mean_spin(
    spin_contriutions: jax.Array, weight: Weight
) -> tuple[jax.Array, Stats]:
    r"""Compute the mean of a batch of spin contributions.

    Args:
        spin_contriutions (jax.Array): the batch of local spin_contributions.
        weight (~deepqmc.types.Weight): the weight of each sample in the batch.

    Returns:
        tuple[jax.Array, ~deepqmc.types.Stats]: a tuple of spin expectation value
        and statistics.
    """
    stats = {
        'spin/mean': jnp.average(spin_contriutions, axis=-1, weights=weight),
        'spin/std': weighted_std(spin_contriutions, axis=-1, weights=weight),
    }
    return all_device_mean(spin_contriutions * weight), stats


def compute_mean_spin_tangent(
    spin_contributions: jax.Array,
    weight: Weight,
    log_psi_tangent: jax.Array,
    gradient_mask: jax.Array,
) -> jax.Array:
    r"""Compute the tangent of the spin with respect to the Ansatz parameters.

    Args:
        spin_contributions (jax.Array): a batch of spin contributions.
        weight (~deepqmc.types.Weight): the weights of each sample in the batch.
        log_psi_tangent (jax.Array): the jvp of the WF values with respect to the Ansatz
            parameters.
        gradient_mask (jax.Array): a boolean samplewise mask to apply to the gradients.

    Returns:
        jax.Array: the jvp of the spin with respect to the Ansatz parameters.
    """
    per_mol_state_mean_spin = all_device_mean(
        spin_contributions * weight, axis=-1, keepdims=True
    )
    spin_contributions_tangent = (
        (spin_contributions - per_mol_state_mean_spin) * log_psi_tangent * weight
    )
    mean_energy_tangent = masked_mean(spin_contributions_tangent, gradient_mask)
    return mean_energy_tangent


def compute_spin_raising_contributions(
    rng: KeyArray,
    hamil: MolecularHamiltonian,
    ansatz: Ansatz,
    phys_conf: PhysicalConfiguration,
    params: Params,
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

    Returns:
        jax.Array: the samplewise contributions to the stochastic spin raising
            expectation value.
    """

    down_idx = jax.random.randint(
        rng, phys_conf.batch_shape, hamil.n_up, hamil.n_up + hamil.n_down
    )
    spin_raising_contributions = jax.vmap(
        jax.vmap(
            jax.vmap(
                make_stochastic_spin_raising_operator(hamil, ansatz.apply), (None, 0, 0)
            )
        ),
        (None, 0, 0),
    )(params, phys_conf, down_idx)
    return spin_raising_contributions


def compute_mean_spin_raising_tangent(
    spin_raising_contributions: jax.Array,
    spin_raising_tangent: jax.Array,
    weight: Weight,
    log_psi_tangent: jax.Array,
    gradient_mask: jax.Array,
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

    Returns:
        jax.Array: the jvp of the spin raising operator with respect to the
            Ansatz parameters.
    """
    per_mol_state_mean_spin_raising = all_device_mean(
        spin_raising_contributions * weight, axis=-1, keepdims=True
    )
    self_adjoint_tangent = (
        spin_raising_contributions - per_mol_state_mean_spin_raising
    ) * log_psi_tangent
    total_tangent = (
        per_mol_state_mean_spin_raising
        * weight
        * (2 * self_adjoint_tangent + spin_raising_tangent)
    )
    mean_total_tangent = masked_mean(total_tangent, gradient_mask)
    return mean_total_tangent
