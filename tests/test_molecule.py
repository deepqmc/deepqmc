import pytest

from deepqmc.molecule import Molecule


@pytest.mark.parametrize('name', ['LiH', 'H2O', 'NH3', 'H10', 'bicyclobutane'])
class TestMolecule:
    def test_from_name(self, name, ndarrays_regression):
        mol = Molecule.from_name(name)
        ndarrays_regression.check(
            {
                'charge': mol.charge,
                'spin': mol.spin,
                'charges': mol.charges,
                'coords': mol.coords,
                'n_nuc': mol.n_nuc,
                'n_up': mol.n_up,
                'n_down': mol.n_down,
                'n_shells': mol.n_shells,
            }
        )
