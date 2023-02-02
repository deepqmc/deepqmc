import pytest

from deepqmc.physics import local_potential, nonlocal_potential


@pytest.mark.parametrize('pp_type', [None, 'bfd', 'ccECP'])
@pytest.mark.parametrize('name', ['LiH', 'C', 'ScO'])
class TestPhysics:
    def test_pseudo_potentials(self, helpers, name, pp_type, ndarrays_regression):
        mol = helpers.mol(name, pp_type)
        hamil = helpers.hamil(mol)
        params, state, paulinet, phys_conf = helpers.create_paulinet(
            hamil, R=helpers.R(name), phys_conf_kwargs={'elec_std': 0.45}
        )
        wf = lambda state, phys_conf: paulinet.apply(params, state, phys_conf)[0]
        ndarrays_regression.check(
            {
                'local_potential': local_potential(phys_conf, mol),
                'nonlocal_potential': (
                    nonlocal_potential(helpers.rng(), phys_conf, mol, state, wf)
                    if pp_type
                    else 0
                ),
            },
            default_tolerance={'rtol': 2e-2, 'atol': 1e-8},
        )
        # note: nonlocal_potential is not particularly
        # numerically stable, hence the large tolerance
