import jax
import jax.numpy as jnp

from ..hamil import MolecularHamiltonian
from ..parallel import all_device_mean
from ..physics import evaluate_spin
from ..types import Ansatz, Params, PhysicalConfiguration, Stats, Weight
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
