import os
import re
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from glob import glob
from importlib import resources
from pathlib import Path
from typing import ClassVar, Optional, cast
from typing_extensions import Self

import jax
import jax.numpy as jnp
import yaml
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import get_original_cwd, to_absolute_path

from .units import angstrom_to_bohr, null

__all__ = ['Molecule']


def mol_conf_dir() -> Path:
    return cast(Path, resources.files('deepqmc').joinpath('conf/hamil/mol'))


def get_all_names() -> set[str]:
    return {filename.replace('.yaml', '') for filename in os.listdir(mol_conf_dir())}


@dataclass(frozen=True, init=False)
class Molecule:
    r"""Represents a molecule.

    The array-like arguments accept anything that can be transformed to
    :class:`jax.Array`.

    Args:
        coords (jax.Array | list[float]):
            nuclear coordinates ((:math:`N_\text{nuc}`, 3), a.u.) as rows
        charges (jax.Array | list[int | float]): atom charges (:math:`N_\text{nuc}`)
        charge (int): total charge of a molecule
        spin (int): total spin multiplicity
        unit (str): units of the coordinates, either 'bohr' or 'angstrom'
        data (dict): additional data stored with the molecule
    """

    all_names: ClassVar[set] = get_all_names()

    coords: jax.Array
    charges: jax.Array
    charge: int
    spin: int
    data: dict

    # DERIVED PROPERTIES:
    n_atom_types: int

    def __init__(
        self,
        *,
        coords,
        charges,
        charge,
        spin,
        unit='bohr',
        data=None,
    ):
        def set_attr(**kwargs):
            for k, v in kwargs.items():
                object.__setattr__(self, k, v)

        unit_multiplier = {'bohr': null, 'angstrom': angstrom_to_bohr}[unit]
        set_attr(
            coords=unit_multiplier(jnp.array(coords)),
            charges=jnp.array(charges, dtype=float),
            charge=charge,
            spin=spin,
            data=data or {},
        )

        # Derived properties
        set_attr(
            n_atom_types=len(jnp.unique(jnp.array(charges))),
        )

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

    @classmethod
    def from_name(cls, name: str) -> Self:
        """Create a molecule from a database of named molecules.

        The available names are in :attr:`Molecule.all_names`.

        Args:
            name (str): name of the molecule (one of :attr:`Molecule.all_names`)
        """
        if name in cls.all_names:
            mol = deepcopy(read_molecule_dataset(mol_conf_dir(), whitelist=name)[name])
        else:
            raise ValueError(f'Unknown molecule name: {name}')
        return mol

    @classmethod
    def from_file(cls, file: str) -> Self:
        """Create a molecule from a YAML file.

        Args:
            file (str): path to the YAML file
        """
        if not Path(file).is_absolute():
            if GlobalHydra().instance().is_initialized():
                file = os.path.join(to_absolute_path(get_original_cwd()), file)
            else:
                file = to_absolute_path(file)
        with open(file, 'r') as stream:
            return cls(**yaml.safe_load(stream))


class MoleculeDict(OrderedDict):
    r"""Store molecules in the order they were added to the dictionary."""

    def __setitem__(self, key: str, value: Molecule):
        super().__setitem__(key, value)
        self.move_to_end(key)


def read_molecule_dataset(
    dataset: Path, whitelist: Optional[str] = None
) -> MoleculeDict:
    molecules = MoleculeDict()
    for f in sorted(glob(str(dataset / '*.yaml'))):
        filename = f.split('/')[-1].replace('.yaml', '')
        if whitelist is not None and not re.search(whitelist, filename):
            continue
        with open(f, 'r') as stream:
            molecules[filename] = Molecule(**yaml.safe_load(stream))
    return molecules
