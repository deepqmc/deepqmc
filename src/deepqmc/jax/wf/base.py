import haiku as hk
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_reduce

__all__ = ['state_callback']


class WaveFunction(hk.Module):
    r"""
    Base class for all trial wave functions.

    Shape:
        - Input, :math:`\mathbf r`, (float, :math:`(N,3)`, a.u.): particle
            coordinates
        - Output1, :math:`\ln|\psi(\mathbf r)|` (float):
        - Output2, :math:`\operatorname{sgn}\psi(\mathbf r)` (float):
    """

    def __init__(self, mol):
        super().__init__()
        self.mol = mol
        self.n_up, self.n_down = mol.n_up, mol.n_down

    @property
    def spin_slices(self):
        return slice(None, self.n_up), slice(self.n_up, None)

    def forward(self, rs):
        return NotImplemented


def state_callback(state, batch_dim=True):
    r"""
    Check haiku state for overflows.

    To avoid frequent recompilations when using a distance cutoff in the
    construction of the molecular graph, the arrays of the graph are allocated
    to have a fixed size. After each iteration, we need to check whether these
    arrays have overflown, and reallocate them with larger sizes if they did.
    This function performs this task, while assuming as little about the
    structure of :data:`state` as possible.

    Args:
        state (dic): the haiku state. A nested dictionary of arrays with shapes
            matching that of the arrays allocated for the graph. The entries in
            each array should indicate the size of that array needed to avoid
            overflows. Unused entries should be zeros. The state dictionary is
            created internally by the haiku library, while the above mentioned
            arrays should be contained in the :data:`occupancies` entry of the
            haiku state.
        batch_dim (bool): optional, whether the state arrays have a first
            batch dimension.
    """
    if not state:
        return state, False
    # Top-level key is the name of the hk.Module that creates the state
    # we assume there is only one such module
    keys = state.keys()
    assert len(keys) == 1
    key = list(keys)[0]

    occupancies = state[key]['occupancies']
    underflow = tree_map(
        lambda x: jnp.any(jnp.logical_and(x < x.shape[-1], x != 0)), occupancies
    )
    underflow = tree_reduce(lambda x, y: x or y, underflow)
    if underflow:
        raise ValueError(
            'some graph edges were dropped due to the distance cutoff, '
            'increase the cutoff distance'
        )
    if batch_dim:
        # Aggregate within batch
        max_occupancy = tree_map(lambda x: jax.numpy.max(x, axis=0), occupancies)
    else:
        max_occupancy = occupancies
    # Aggregate over batches
    max_occupancy = tree_map(lambda x: jax.numpy.max(x, axis=-1).item(), max_occupancy)

    overflow_per_entry = tree_map(
        lambda occ, max_occ: (occ.shape[-1] < max_occ), occupancies, max_occupancy
    )
    overflow = tree_reduce(lambda x, y: x or y, overflow_per_entry)

    def create_new_occupancies(old, shape):
        if shape > old.shape[-1]:
            new = jax.numpy.zeros_like(old, shape=(*old.shape[:-1], shape))
            copy_len = min(old.shape[-1], new.shape[-1])
            return new.at[..., :copy_len].set(old[..., :copy_len])
        else:
            return old

    occupancies = tree_map(create_new_occupancies, occupancies, max_occupancy)

    return {key: {'occupancies': occupancies}}, overflow
