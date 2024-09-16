import jax

from ..hamil import MolecularHamiltonian
from ..parallel import all_device_mean
from ..types import (
    Energy,
    KeyArray,
    ParametrizedWaveFunction,
    Params,
    PhysicalConfiguration,
    Stats,
    Weight,
)
from ..utils import masked_mean


def compute_local_energy(
    rng: KeyArray,
    hamil: MolecularHamiltonian,
    ansatz: ParametrizedWaveFunction,
    params: Params,
    phys_conf: PhysicalConfiguration,
) -> tuple[Energy, Stats]:
    r"""Compute a batch of local energies.

    Args:
        rng (~deepqmc.types.KeyArray): rng key to use for the generation of the ECP
            quadratures.
        hamil (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the system.
        ansatz (~deepqmc.types.ParametrizedWaveFunction): the parametrized
            wave function.
        params (~deepqmc.types.Params): the current parameters of the Ansatz.
        phys_conf (~deepqmc.types.PhysicalConfiguration): a batch of input to the
            Ansatz.

    Returns:
        Tuple[~deepqmc.types.Energy, ~deepqmc.types.Stats]: a tuple of local energy and
            statistics.
    """
    rng = jax.random.split(rng, phys_conf.batch_shape)

    local_energy, hamil_stats = jax.vmap(  # molecule_batch
        jax.vmap(  # electronic_state
            jax.vmap(hamil.local_energy(ansatz), (0, None, 0))  # electron_batch
        ),
        (0, None, 0),
    )(rng, params, phys_conf)

    stats = jax.tree_util.tree_map(lambda x: x.mean(axis=-1), hamil_stats)
    return local_energy, stats


def compute_mean_energy(local_energy: Energy, weight: Weight) -> tuple[Energy, Stats]:
    r"""Compute the mean of a batch of local energies.

    Args:
        local_energy (~deepqmc.types.Energy): the batch of local energies.
        weight (~deepqmc.types.Weight): the weight of each sample in the batch.

    Returns:
        Tuple[~deepqmc.types.Energy, ~deepqmc.types.Stats]: a tuple of mean energy and
            statistics.
    """
    return all_device_mean(local_energy * weight), {}


def compute_mean_energy_tangent(
    local_energy: Energy,
    weight: Weight,
    log_psi_tangent: jax.Array,
    gradient_mask: jax.Array,
) -> jax.Array:
    r"""Compute the tangent of the mean energy with respect to the Ansatz parameters.

    Args:
        local_energy (~deepqmc.types.Energy): a batch of local energies.
        weight (~deepqmc.types.Weight): the weights of each sample in the batch.
        log_psi_tangent (jax.Array): the jvp of the WF values with respect to the Ansatz
            parameters.
        gradient_mask (jax.Array): a boolean samplewise mask to apply to the gradients.

    Returns:
        jax.Array: the jvp of the mean energy with respect to the Ansatz parameters.
    """
    per_mol_state_mean_energy = all_device_mean(
        local_energy * weight, axis=-1, keepdims=True
    )
    local_energy_tangent = (
        (local_energy - per_mol_state_mean_energy) * log_psi_tangent * weight
    )
    mean_energy_tangent = masked_mean(local_energy_tangent, gradient_mask)
    return mean_energy_tangent
