import pytest


@pytest.mark.parametrize('ecp_type', [None, 'bfd', 'ccECP'])
@pytest.mark.parametrize('name', ['LiH', 'C'])
class TestPhysics:
    def test_pseudo_potentials(self, helpers, name, ecp_type, ndarrays_regression):
        mol = helpers.mol(name)
        hamil = helpers.hamil(mol, ecp_type)
        phys_conf = helpers.phys_conf(hamil)
        _wf, params = helpers.create_ansatz(hamil)
        wf = lambda phys_conf: _wf.apply(params, phys_conf)
        ndarrays_regression.check(
            {
                'local_potential': hamil.potential.local_potential(phys_conf),
                'nonlocal_potential': (
                    hamil.potential.nonloc_potential(helpers.rng(), phys_conf, wf)
                    if ecp_type
                    else 0
                ),
            },
        )
        # note: nonlocal_potential is not particularly
        # numerically stable, hence the large tolerance
