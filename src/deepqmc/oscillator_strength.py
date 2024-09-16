from typing import Optional

import jax
import jax.numpy as jnp

from .types import Energy


def compute_oscillator_strength(
    local_energies: Energy,
    ratios: jax.Array,
    rs: jax.Array,
    local_energies_mask: Optional[jax.Array] = None,
    ratios_mask: Optional[jax.Array] = None,
):
    """Computes the oscillator strength and error for a batch.

    Args:
    local_energies (jax.Array): the electron batch of local energies, shape:
        ``[electronic_states, electron_batch_size]``.
    ratios (jax.Array): the electron batch of wave function ratios, shape:
        ``[electronic_states, electronic_states, electron_batch_size]``.
    rs (jax.Array): the electron batch of electron samples, shape:
        ``[electronic_states, electron_batch_size, n_electrons, 3]``.

    local_energies_mask (jax.Array): Optional, mask for the local energies.
    ratios_mask (jax.Array): Optional, mask for the ratios.
    """

    sample_size = local_energies.shape[-1]

    # excitation energy
    energy_mean = jnp.mean(local_energies, axis=-1, where=local_energies_mask)
    energy_err = (
        jnp.std(local_energies, axis=-1, where=local_energies_mask) / sample_size**0.5
    )
    ex_energy_mean = energy_mean[None, :] - energy_mean[:, None]
    ex_energy_err = (energy_err**2 + energy_err[:, None] ** 2) ** 0.5

    # dipole strength
    cd = jnp.sum(-rs, axis=-2) * ratios[..., None]
    ratios_mask = ratios_mask if ratios_mask is None else ratios_mask[..., None]
    cd_mean = jnp.mean(cd, axis=-2, where=ratios_mask)
    cd_err = jnp.std(cd, axis=-2, where=ratios_mask) / sample_size**0.5
    cd_rel_err = cd_err / cd_mean

    ds_vec = cd_mean * cd_mean.swapaxes(0, 1)
    ds_err_vec = (
        jnp.abs(ds_vec) * (cd_rel_err**2 + cd_rel_err.swapaxes(0, 1) ** 2) ** 0.5
    )

    ds_mean = jnp.sum(ds_vec, axis=-1)
    ds_err = jnp.sum(ds_err_vec**2, axis=-1) ** 0.5

    # transition dipole moment
    tdm_mean = ds_mean**0.5
    tdm_err = 0.5 * tdm_mean * (ds_err / ds_mean)

    # oscillator strength
    os_mean = (2 / 3) * ex_energy_mean * ds_mean
    os_err = (
        (2 / 3)
        * jnp.abs(os_mean)
        * ((ex_energy_err / ex_energy_mean) ** 2 + (ds_err / ds_mean) ** 2) ** 0.5
    )

    return (os_mean, os_err), (tdm_mean, tdm_err), (ex_energy_mean, ex_energy_err)
