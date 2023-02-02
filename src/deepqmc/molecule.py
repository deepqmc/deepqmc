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
            system = yaml.safe_load(stream)
            data[f.strip('.yaml')] = system
    return data


_SYSTEMS = parse_molecules()


@dataclass(frozen=True, init=False)
class Molecule:
    r"""Represents a molecule, defined by atomic numbers, charge and spin.

    The :data:`charges` argument accepts anything that can be transformed to
    :class:`jax.numpy.DeviceArray`.

    Args:
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
    pp_nl_params: jnp.ndarray

    # REDUNDANT AUXILIARY PROPERTIES:
    # True if at least one nucleus uses pseudopotential
    any_pp: bool
    # Indices of nuclei with pseudopotential
    nuc_with_nl_pot: jnp.ndarray

    def __init__(
        self,
        *,
        charges,
        charge,
        spin,
        coords=None,
        unit=None,
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

        set_attr(
            charges=1.0 * jnp.asarray(charges),
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
            nuc_with_nl_pot=jnp.nonzero(self.pp_mask)[0],
        )

        shells = [get_shell(z) for z in self.charges]
        pp_shells = [get_shell(z + 1) - 1 for z in self.ns_core]
        set_attr(n_shells=tuple(shells))
        set_attr(n_pp_shells=tuple(pp_shells))

    def __len__(self):
        return len(self.charges)

    def __iter__(self):
        yield from self.charges

    def __repr__(self):
        return (
            'Molecule(\n'
            f'  charges={self.charges},\n'
            f'  charge={self.charge},\n'
            f'  spin={self.spin}\n'
            f'  ns_core={self.ns_core}\n'
            ')'
        )

    def as_pyscf(self, R):
        return [(int(charge), coord) for coord, charge in zip(R, self)]

    @property
    def n_particles(self):
        r"""Return the number of nuclei, spin-up, and spin-down electrons."""
        return self.n_nuc, self.n_up, self.n_down

    @classmethod
    def from_name(cls, name, pp_type=None, pp_mask=None, **kwargs):
        """Create a molecule from a database of named molecules.

        The available names are in :attr:`Molecule.all_names`.
        """
        if name in cls.all_names:
            system = deepcopy(_SYSTEMS[name])
            system.update(kwargs)
            system.pop('coords')
            system.pop('unit', None)
        else:
            raise ValueError(f'Unknown molecule name: {name}')
        return cls(**system, pp_type=pp_type, pp_mask=pp_mask)

    @classmethod
    def default_coords_from_name(cls, name):
        """Return the nuclear coordinates stored in the config files."""
        if name in cls.all_names:
            system = deepcopy(_SYSTEMS[name])
            unit_multiplier = {'bohr': 1.0, 'angstrom': angstrom}[
                system.get('unit', 'bohr')
            ]
            coords = jnp.asarray(system['coords']) * unit_multiplier
        else:
            raise ValueError(f'Unknown molecule name: {name}')
        return coords
