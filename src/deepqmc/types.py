from collections import namedtuple

import jax.numpy as jnp
import jax_dataclasses as jdc

Psi = namedtuple('Psi', 'sign log')
TrainState = namedtuple('TrainState', 'sampler params opt')


@jdc.pytree_dataclass
class PhysicalConfiguration:
    r"""Represent input physical configurations of electrons and nuclei.

    It currently contains the nuclear and electronic coordinates, along with
    :data:`mol_idx`, which specifies which nuclear configuration a given sample
    was obtained from.
    """

    R: jnp.ndarray
    r: jnp.ndarray
    mol_idx: jnp.ndarray

    def __getitem__(self, idx):
        return self.__class__(
            self.R.__getitem__(idx),
            self.r.__getitem__(idx),
            self.mol_idx.__getitem__(idx),
        )

    def __len__(self):
        return len(self.r)

    @property
    def batch_shape(self):
        return self.r.shape[:-2]


class Potential:
    r"""Base class for the (pseudo)potential in which electrons move.

    Does not include the electron-electron repulsion.
    """

    def local_potential(self, phys_conf):
        r"""Compute the (local pseudo)potential energy of the electrons.

        Args:
            phys_conf (:class:`deepqmc.types.PhysicalConfiguration`): electron and
                nuclear coordinates.
        """
        raise NotImplementedError

    def nonloc_potential(self, rng, phys_conf, wf):
        return 0.0
