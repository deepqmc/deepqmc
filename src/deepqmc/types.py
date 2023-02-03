from collections import namedtuple

import jax.numpy as jnp
import jax_dataclasses as jdc

Psi = namedtuple('Psi', 'sign log')


@jdc.pytree_dataclass
class PhysicalConfiguration:
    r"""Represent input physical configurations of electrons and nuclei.

    It currently contains the nuclear and electronic coordinates, along with
    :data:`config_idx`, which specifies which nuclear configuration a given sample
    was obtained from.
    """

    R: jnp.ndarray
    r: jnp.ndarray
    config_idx: jnp.ndarray

    def __getitem__(self, idx):
        return self.__class__(
            self.R.__getitem__(idx),
            self.r.__getitem__(idx),
            self.config_idx.__getitem__(idx),
        )

    def __len__(self):
        return len(self.r)
