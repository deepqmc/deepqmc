import jax
import jax.numpy as jnp

from ..types import KeyArray, SamplerState, Stats


class IdleNucleiSampler:
    r"""
    Keeps track of nuclei without updating positions.

    Args:
        nuc_coords (jax.Array): initial coordinates of the sampled molecules


    """

    def __init__(self, charges: jax.Array):
        pass

    def init(self, nuc_coords: jax.Array, *args, **kwargs) -> SamplerState:
        state = {'R': nuc_coords}
        return state

    def sample(
        self, rng: KeyArray, state: SamplerState
    ) -> tuple[SamplerState, jax.Array, Stats]:
        return state, jnp.zeros_like(state['R']), {}


def no_elec_warp(
    rng: KeyArray, R: jax.Array, dR: jax.Array, smpl_state: SamplerState
) -> SamplerState:
    r"""Identity electron warp function."""
    return smpl_state
