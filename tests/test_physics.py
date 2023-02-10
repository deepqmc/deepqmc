import pytest

from deepqmc.physics import local_potential, nonlocal_potential


@pytest.mark.parametrize('pp_type', [None, 'bfd', 'ccECP'])
@pytest.mark.parametrize('name', ['LiH', 'C', 'ScO'])
class TestPhysics:
    def test_pseudo_potentials(self, helpers, name, pp_type, ndarrays_regression):
        mol = helpers.mol(name, pp_type)
        params, state, paulinet, rs = helpers.create_paulinet(
            hamil=helpers.hamil(mol, elec_std=0.45)
        )
        wf = lambda state, r: paulinet.apply(params, state, r)[0]
        ndarrays_regression.check(
            {
                'local_potential': local_potential(rs, mol),
                'nonlocal_potential': nonlocal_potential(
                    helpers.rng(), rs, mol, state, wf
                ),
            },
            default_tolerance={'rtol': 2e-2, 'atol': 1e-8},
        )
        # note: nonlocal_potential is not particularly
        # numerically stable, hence the large tolerance
