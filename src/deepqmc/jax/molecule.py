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

        n_elec = int(self.charges.sum()) - self.charge
        self.n_up = (n_elec + self.spin) // 2
        self.n_down = n_elec - self.n_up
        self.n_shells = self._n_shells()

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

    def _n_shells(self):
        shells = []
        for z in self.charges:
            # find number of occupied shells for atom
            max_elec = 0
            n_shells = 0
            for n in range(10):
                if z <= max_elec:
                    break
                else:
                    n_shells += 1
                    for m in range(n + 1):
                        max_elec += 2 * (2 * m + 1)
            # adding the lowest unoccupied shell might be beneficial,
            # especially for transition metals
            #  n_shells += 1
            shells.append(n_shells)
        return tuple(shells)

    def as_pyscf(self):
        return [(int(charge), coord) for coord, charge in self]

    def n_particles(self):
        """Returns the number of nuclei, spin-up, and spin-down electrons
        of the molecule.
        """
        n_nuclei = len(self.charges)
        n_elec = int(self.charges.sum()) - self.charge
        n_up = (n_elec + self.spin) // 2
        n_down = n_elec - n_up
        return n_nuclei, n_up, n_down
