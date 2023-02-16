import os
from copy import deepcopy
from dataclasses import dataclass
from importlib import resources
from itertools import count
from typing import ClassVar

import jax.numpy as jnp
import yaml

from deepqmc.wf.baseline.pyscfext import parse_pp_params

angstrom = 1 / 0.52917721092

__all__ = ['Molecule']


def get_shell(z):
    # returns the number of (at least partially) occupied shells for 'z' electrons
    # 'get_shell(z+1)-1' yields the number of fully occupied shells for 'z' electrons
    max_elec = 0
    for n in count():
        if z <= max_elec:
            break
        max_elec += 2 * (1 + n) ** 2
    return n


def parse_molecules():
    path = resources.files('deepqmc').joinpath('conf/hamil/mol')
    data = {}
    for f in os.listdir(path):
        with open(path.joinpath(f), 'r') as stream:
            data[f.strip('.yaml')] = yaml.safe_load(stream)
    return data


_SYSTEMS = parse_molecules()


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
        pp_type (str): If set, use the appropriate pseudopotential. The string is passed
            to :func:`pyscf.gto.M()` as :data:`'ecp'` argument. Currently supported
            pseudopotential types: :data:`'bfd'` [Burkatzki et al. 2007],
            :data:`'ccECP'` [Bennett et al. 2017]. Other types might not work properly.
        pp_mask (list, (:math:`N_\text{nuc}`)): list of True and False values specifying
            whether to use a pseudopotential for each nucleus
    """

    all_names: ClassVar[set] = set(_SYSTEMS.keys())

    coords: jnp.ndarray
    charges: jnp.ndarray
    charge: int
    spin: int
    pp_mask: jnp.ndarray  # list of bools
    pp_type: str
    data: dict

    # DERIVED PROPERTIES:
    n_nuc: int
    n_atom_types: int
    n_up: int
    n_down: int
    # total numbers of occupied shells
    n_shells: tuple
    # number of shells fully occupied by 'ns_core' inner electrons (that are replaced
    # by pseudopotential)
    n_pp_shells: tuple
    # number of core electrons replaced by the pseudopotential for each core
    ns_core: jnp.ndarray
    # number of valence electrons for each nucleus (for neutral molecule) or a total
    # number of valence slots in case of charged molecule
    ns_valence: jnp.ndarray
    # stores the parameters of local potential (loaded from [Burkatzki et al. 2007])
    pp_loc_params: jnp.ndarray
    # stores the parameters of non-local potential part
    pp_nl_params: list
    # True if at leas one nucleus uses pseudopotential
    any_pp: bool

    def __init__(
        self,
        *,
        coords,
        charges,
        charge,
        spin,
        unit='bohr',
        data=None,
        pp_type=None,
        pp_mask=None,
    ):
        def set_attr(**kwargs):
            for k, v in kwargs.items():
                object.__setattr__(self, k, v)

        if pp_type is None:
            pp_mask = [False] * len(charges)
        elif pp_mask is None:
            pp_mask = [True] * len(charges)

        assert len(pp_mask) == len(charges), "Incompatible shape of 'pp_mask' given!"

        unit_multiplier = {'bohr': 1.0, 'angstrom': angstrom}[unit]
        set_attr(
            coords=unit_multiplier * jnp.asarray(coords),
            charges=1.0 * jnp.asarray(charges, dtype=jnp.int8),
            charge=charge,
            spin=spin,
            pp_mask=jnp.array(pp_mask),
            pp_type=pp_type,
            data=data or {},
        )

        # Derived properties
        ns_core, pp_loc_params, pp_nl_params = parse_pp_params(self)

        n_elec = int(sum(charges) - sum(ns_core) - charge)
        assert not (n_elec + spin) % 2
        set_attr(
            n_nuc=len(charges),
            n_atom_types=len(jnp.unique(jnp.asarray(charges))),
            n_up=(n_elec + spin) // 2,
            n_down=(n_elec - spin) // 2,
            ns_valence=self.charges - ns_core,
            ns_core=ns_core,
            pp_loc_params=pp_loc_params,
            pp_nl_params=pp_nl_params,
            any_pp=any(self.pp_mask),
        )

        shells = [get_shell(z) for z in self.charges]
        pp_shells = [get_shell(z + 1) - 1 for z in self.ns_core]
        set_attr(n_shells=tuple(shells))
        set_attr(n_pp_shells=tuple(pp_shells))

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
            f'  ns_core={self.ns_core}\n'
            ')'
        )

    def as_pyscf(self):
        return [(int(charge), coord) for coord, charge in self]

    @property
    def n_particles(self):
        r"""Return the number of nuclei, spin-up, and spin-down electrons."""
        return self.n_nuc, self.n_up, self.n_down

    @classmethod
    def from_name(cls, name, **kwargs):
        """Create a molecule from a database of named molecules.

        The available names are in :attr:`Molecule.all_names`.
        """
        if name in cls.all_names:
            system = deepcopy(_SYSTEMS[name])
            system.update(kwargs)
        else:
            raise ValueError(f'Unknown molecule name: {name}')
        coords = system.pop('coords')
        return cls(coords=coords, **system)
