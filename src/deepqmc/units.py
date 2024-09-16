from scipy import constants


def bohr_to_angstrom(length_bohr):
    """Converts bohrs to angstroms."""

    return length_bohr * constants.value('atomic unit of length') / constants.angstrom


def angstrom_to_bohr(length_angstrom):
    """Converts angstroms to bohrs."""

    return (
        length_angstrom * constants.angstrom / constants.value('atomic unit of length')
    )


def eV_to_hartree(energy_eV):
    """Converts electron volts to hartrees."""

    return energy_eV * constants.value('electron volt-hartree relationship')


def eV_to_kcal_mol(energy_eV):
    """Converts electron volts to kcals/mol."""

    return energy_eV * constants.eV / (constants.calorie * 1e3 / constants.N_A)


def hartree_to_eV(energy_hartree):
    """Converts hartrees to electron volts."""

    return energy_hartree / constants.value('electron volt-hartree relationship')


def hartree_to_kcal_mol(energy_hartree):
    """Converts hartrees to kcals/mol."""

    return (
        energy_hartree
        * constants.value('Hartree energy')
        / (constants.calorie * 1e3 / constants.N_A)
    )


def kcal_mol_to_hartree(energy_kcal_mol):
    """Converts kcals/mol to hartrees."""

    return (
        energy_kcal_mol
        * (constants.calorie * 1e3 / constants.N_A)
        / constants.value('Hartree energy')
    )


def kcal_mol_to_eV(energy_kcal_mol):
    """Converts kcals/mol to electron volts."""

    return energy_kcal_mol * (constants.calorie * 1e3 / constants.N_A) / constants.eV


def null(anything):
    """Returns unit without conversion."""
    return anything
