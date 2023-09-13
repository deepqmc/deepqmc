import pytest

from deepqmc.molecule import Molecule


@pytest.mark.parametrize(
    'name',
    [
        'LiH',
        'H2O',
        'NH3',
        'H10',
        'bicyclobutane',
        'C',
        'ScO',
    ],
)
class TestMolecule:
    def test_from_name(self, name, ndarrays_regression):
        mol = Molecule.from_name(name)
        ndarrays_regression.check(
            {
                'charge': mol.charge,
                'spin': mol.spin,
                'charges': mol.charges,
                'coords': mol.coords,
            }
        )
