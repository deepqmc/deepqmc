import haiku as hk
import jax

__all__ = ['state_callback']


class WaveFunction(hk.Module):
    r"""Base class for all trial wave functions.
    Shape:
        - Input, :math:`\mathbf r`, a.u.: :math:`(\cdot,N,3)`
        - Output1, :math:`\ln|\psi(\mathbf r)|`: :math:`(\cdot)`
        - Output2, :math:`\operatorname{sgn}\psi(\mathbf r)`: :math:`(\cdot)`
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
        state (dic): the haiku state. A dictionary of arrays with shapes matching
            that of the arrays allocated for the graph. The entries in each array
            should indicate the size of that array needed to avoid overflows.
            Unused entries should be zeros.
        batch_dim (bool): optional, whether the state arrays have a first
            batch dimension.
    """
    if not state:
        return state, False
    key = list(state.keys())[0]
    occupancies = state[key]['occupancies']
    n_occupancies = state[key]['n_occupancies']
    if batch_dim:
        # Aggregate within batch
        new_shape = jax.tree_util.tree_map(
            lambda x: jax.numpy.max(x, axis=0), occupancies
        )
    else:
        new_shape = occupancies
    # Aggregate over batches
    new_shape = jax.tree_util.tree_map(
        lambda x: jax.numpy.max(x, axis=-1).item(), new_shape
    )
    larger = jax.tree_util.tree_map(
        lambda old, new_shape: (old.shape[-1] < new_shape), occupancies, new_shape
    )
    overflow = jax.tree_util.tree_reduce(lambda x, y: x or y, larger)

    def create_new_occupancies(old, shape):
        if shape > old.shape[-1]:
            return jax.numpy.zeros_like(old, shape=(*old.shape[:-1], shape))
        else:
            return old

    new_occupancies = jax.tree_util.tree_map(
        create_new_occupancies, occupancies, new_shape
    )

    def copy_state(old, new):
        copy_len = min(old.shape[-1], new.shape[-1])
        return new.at[..., :copy_len].set(old[..., :copy_len])

    return {
        key: {
            'n_occupancies': n_occupancies,
            'occupancies': jax.tree_util.tree_map(
                copy_state, occupancies, new_occupancies
            ),
        }
    }, overflow
