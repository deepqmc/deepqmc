import jax.numpy as jnp

angstrom = 1 / 0.52917721092


class Molecule:
    """Represents a molecule.
    The array-like arguments accept anything that can be transformed to
    :class:`torch.Tensor`.
    Args:
        coords (array-like, rank-2, float, a.u.): atom coordinates as rows
        charges (array-like, rank-1, float): atom charges
        charge (int): total charge of a molecule
        spin (int): total spin multiplicity
    """

    # all_names = set(_SYSTEMS.keys())

    def __init__(self, coords, charges, charge, spin, unit='bohr', data=None):
        assert len(coords) == len(charges)
        super().__init__()
        unit_multiplier = {'bohr': 1.0, 'angstrom': angstrom}[unit]
        self.coords = unit_multiplier * jnp.asarray(coords)
        self.charges = 1.0 * jnp.asarray(charges)
        self.charge = charge
        self.spin = spin
        self.data = data or {}

    def __len__(self):
        return len(self.charges)

    def __iter__(self):
        yield from zip(self.coords, self.charges)

    def __repr__(self):
        return (
            'Molecule(\n'
            f'  coords={self.coords},\n'
            f'  charges={self.charges},\n'
            f'  charge={self.charge},\n'
            f'  spin={self.spin}\n'
            ')'
        )

    def as_pyscf(self):
        return [(int(charge), coord) for coord, charge in self]
