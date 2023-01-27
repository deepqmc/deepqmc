import pytest

from deepqmc.molecule import Molecule


@pytest.mark.parametrize(
    'name,pp_type',
    [
        ('LiH', None),
        ('H2O', None),
        ('NH3', None),
        ('H10', None),
        ('bicyclobutane', None),
        ('C', 'ccECP'),
        ('ScO', 'ccECP'),
        ('ScO', 'bfd'),
    ],
)
class TestMolecule:
    def test_from_name(self, name, pp_type, ndarrays_regression):
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
                'n_pp_shells': mol.n_pp_shells,
                'ns_core': mol.ns_core,
                'ns_valence': mol.ns_valence,
                'pp_loc_params': mol.pp_loc_params,
                'pp_nl_params': mol.pp_nl_params,
            }
        )
