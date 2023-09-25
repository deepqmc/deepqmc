import pytest


@pytest.mark.parametrize('pp_type', [None, 'bfd', 'ccECP'])
@pytest.mark.parametrize('name', ['LiH', 'C'])
class TestPhysics:
    def test_pseudo_potentials(self, helpers, name, pp_type, ndarrays_regression):
        mol = helpers.mol(name)
        hamil = helpers.hamil(mol, pp_type)
        phys_conf = helpers.phys_conf(hamil)
        _wf, params = helpers.create_ansatz(hamil)
        wf = lambda phys_conf: _wf.apply(params, phys_conf)
        ndarrays_regression.check(
            {
                'local_potential': hamil.potential.local_potential(phys_conf),
                'nonlocal_potential': (
                    hamil.potential.nonloc_potential(helpers.rng(), phys_conf, wf)
                    if pp_type
                    else 0
                ),
            },
            default_tolerance={'rtol': 2e-2, 'atol': 1e-8},
        )
        # note: nonlocal_potential is not particularly
        # numerically stable, hence the large tolerance
