import pytest

from deepqmc.ecp.composite_potential import CompositePotential
from deepqmc.ecp.gaussian_type_ecp import GaussianTypeECP
from deepqmc.ecp.pseudo_hamiltonian import PseudoHamiltonian
from deepqmc.hamil import MolecularHamiltonian
from deepqmc.molecule import Molecule
from deepqmc.physics import NuclearCoulombPotential


@pytest.mark.parametrize(
    'hamil,ecp_type',
    [
        (MolecularHamiltonian, None),
        (MolecularHamiltonian, 'ccECP'),
    ],
    ids=['Molecular', 'Molecular+PP'],
)
class TestHamil:
    SAMPLE_SIZE = 5

    def test_init(self, helpers, hamil, ecp_type, ndarrays_regression):
        hamil = helpers.hamil(helpers.mol(), ecp_type=ecp_type)
        ndarrays_regression.check(
            {
                'n_up': hamil.n_up,
                'n_down': hamil.n_down,
                'ns_valence': hamil.ns_valence,
                'pp_mask': hamil.ecp_mask,
            }
        )

    def test_init_sample(self, helpers, hamil, ecp_type, ndarrays_regression):
        hamil = helpers.hamil(helpers.mol(), ecp_type=ecp_type)
        phys_conf = helpers.phys_conf(hamil, n=self.SAMPLE_SIZE)
        ndarrays_regression.check({'rs': phys_conf.r})

    def test_local_energy(self, helpers, hamil, ecp_type, ndarrays_regression):
        hamil = helpers.hamil(helpers.mol(), ecp_type=ecp_type)
        phys_conf = helpers.phys_conf(hamil)
        ansatz, params = helpers.create_ansatz(hamil)
        E_loc, _ = hamil.local_energy(ansatz.apply)(helpers.rng(), params, phys_conf)
        ndarrays_regression.check(
            {'E_loc': E_loc},
            default_tolerance={'rtol': 2e-4},
        )


class TestPerElementEcpType:
    @pytest.fixture
    def mol(self):  # ClF2
        return Molecule(
            coords=[[0.0, 0.0, 0.0], [3.08, 0.0, 0.0], [-3.08, 0.0, 0.0]],
            charges=[17, 9, 9],
            charge=0,
            spin=1,
            unit='bohr',
        )

    @pytest.mark.parametrize(
        'ecp_type,ecp_mask,ecp_types,potential',
        [
            (None, None, [None, None, None], NuclearCoulombPotential),
            ('ccECP', None, ['ccECP'] * 3, GaussianTypeECP),
            ('PHcc', [True, False, False], ['PHcc', None, None], PseudoHamiltonian),
            ({'F': 'ccECP'}, None, [None, 'ccECP', 'ccECP'], GaussianTypeECP),
            ({17: 'PHcc'}, None, ['PHcc', None, None], PseudoHamiltonian),
            (
                {'Cl': 'PHcc', 9: 'ccECP', 'Fe': 'PHcc'},
                None,
                ['PHcc', 'ccECP', 'ccECP'],
                CompositePotential,
            ),
        ],
        ids=['none', 'string', 'string+mask', 'symbol', 'number', 'mixed'],
    )
    def test_init(self, mol, ecp_type, ecp_mask, ecp_types, potential):
        hamil = MolecularHamiltonian(mol=mol, ecp_type=ecp_type, ecp_mask=ecp_mask)
        assert hamil.ecp_types == ecp_types
        assert isinstance(hamil.pot, potential)
        assert hamil.ecp_mask.tolist() == [t is not None for t in ecp_types]

    @pytest.mark.parametrize(
        'ecp_type,ecp_mask,match',
        [
            ({'Cl': 'PHcc'}, [True, False, False], 'cannot be used'),
            ({'Cl': 'PHcc', 17: 'PHhf'}, None, 'more than once'),
            ({'Cl': 'bfd', 'F': 'ccECP'}, None, 'Only one Gaussian-type ECP'),
        ],
        ids=['mapping+mask', 'element-twice', 'two-ecp-types'],
    )
    def test_invalid(self, mol, ecp_type, ecp_mask, match):
        with pytest.raises(AssertionError, match=match):
            MolecularHamiltonian(mol=mol, ecp_type=ecp_type, ecp_mask=ecp_mask)

    def test_as_pyscf(self, mol):
        hamil = MolecularHamiltonian(mol=mol, ecp_type={'Cl': 'PHcc', 'F': 'ccECP'})
        # pyscf does not provide PHs, they are replaced by ccECP
        assert hamil.as_pyscf()['ecp'] == {17: 'ccECP', 9: 'ccECP'}
