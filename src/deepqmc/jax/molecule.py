from dataclasses import dataclass
from itertools import count

import jax.numpy as jnp

angstrom = 1 / 0.52917721092

__all__ = ['Molecule']


@dataclass(frozen=True, init=False)
class Molecule:
    r"""Represents a molecule.

    The array-like arguments accept anything that can be transformed to
    :class:`jax.numpy.DeviceArray`.

    Args:
        coords (float, (:math:`N_\text{nuc}`, 3), a.u.):
            nuclear coordinates as rows
        charges (int, (:math:`N_\text{nuc}`)): atom charges
        charge (int): total charge of a molecule
        spin (int): total spin multiplicity
    """

    coords: jnp.ndarray
    charges: jnp.ndarray
    charge: int
    spin: int
    data: dict

    # Derived properties
    n_nuc: int
    n_up: int
    n_down: int
    n_shells: tuple

    # TODO
    # all_names = set(_SYSTEMS.keys())

    def __init__(self, coords, charges, charge, spin, unit='bohr', data=None):
        def set_attr(**kwargs):
            for k, v in kwargs.items():
                object.__setattr__(self, k, v)

        unit_multiplier = {'bohr': 1.0, 'angstrom': angstrom}[unit]
        set_attr(
            coords=unit_multiplier * jnp.asarray(coords),
            charges=1.0 * jnp.asarray(charges),
            charge=charge,
            spin=spin,
            data=data or {},
        )

        # Derived properties
        n_elec = int(sum(charges) - charge)
        assert not (n_elec + spin) % 2
        set_attr(
            n_nuc=len(charges),
            n_up=(n_elec + spin) // 2,
            n_down=(n_elec - spin) // 2,
        )
        shells = []
        for z in charges:
            max_elec = 0
            for n in count():
                if z <= max_elec:
                    break
                max_elec += 2 * (1 + n) ** 2
            shells.append(n)
        set_attr(n_shells=tuple(shells))

    def __len__(self):
        return len(self.charges)

    def __iter__(self):
        yield from zip(self.coords, self.charges)

    def __repr__(self):
        return (
            'Molecule(\n'
            f'  coords=\n{self.coords},\n'
            f'  charges={self.charges},\n'
            f'  charge={self.charge},\n'
            f'  spin={self.spin}\n'
            ')'
        )

    def as_pyscf(self):
        return [(int(charge), coord) for coord, charge in self]

    @property
    def n_particles(self):
        r"""Return the number of nuclei, spin-up, and spin-down electrons."""
        return len(self), self.n_up, self.n_down
