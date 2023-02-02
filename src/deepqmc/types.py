from collections import namedtuple

import jax.numpy as jnp
from jax_dataclasses import pytree_dataclass

Psi = namedtuple('Psi', 'sign log')


@pytree_dataclass
class PhysicalConfiguration:
    r"""Represent input physical configurations of electrons and nuclei.

    It currently contains the nuclear and electronic coordinates, along with
    :data:`config_idx`, which specifies which nuclear configuration a given sample
    was obtained from.
    """

    R: jnp.ndarray
    r: jnp.ndarray

    def __getitem__(self, idx):
        return self.__class__(
            self.R.__getitem__(idx),
            self.r.__getitem__(idx),
        )
