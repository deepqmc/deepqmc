import pytest
import torch
from torch import nn

from deepqmc import Molecule
from deepqmc.fit import LossEnergy, fit_wf
from deepqmc.physics import local_energy
from deepqmc.sampling import LangevinSampler
from deepqmc.wf import PauliNet
from deepqmc.wf.paulinet.distbasis import DistanceBasis
from deepqmc.wf.paulinet.gto import GTOBasis
from deepqmc.wf.paulinet.schnet import ElectronicSchNet

try:
    import pyscf.gto
except ImportError:
    pyscf_marks = [pytest.mark.skip(reason='Pyscf not installed')]
else:
    pyscf_marks = []


def assert_alltrue_named(items):
    dct = dict(items)
    assert dct == {k: True for k in dct}


@pytest.fixture
def rs():
    return torch.randn(5, 3, 3)


@pytest.fixture
def mol():
    mol = Molecule.from_name('H2')
    mol.charge = -1
    mol.spin = 3
    return mol


@pytest.fixture(params=[pytest.param(PauliNet, marks=pyscf_marks)])
def net_factory(request):
    return request.param


class OmniNet(nn.Module):
    def __init__(self, n_atoms, n_up, n_down, n_orbitals, n_backflows):
        super().__init__()
        self.dist_basis = DistanceBasis(4, envelope='nocusp')
        self.schnet = ElectronicSchNet(
            n_up,
            n_down,
            n_atoms,
            dist_feat_dim=4,
            n_interactions=2,
            kernel_dim=8,
            embedding_dim=16,
            version=1,
        )
        self.orbital = nn.Linear(16, 1, bias=False)

    def forward(self, dists_nuc, dists_elec):
        edges_nuc = self.dist_basis(dists_nuc)
        edges_elec = self.dist_basis(dists_elec)
        xs = self.schnet(edges_elec, edges_nuc)
        return self.orbital(xs).squeeze(dim=-1).sum(dim=-1), None


@pytest.fixture
def wf(net_factory, mol):
    args = (mol,)
    kwargs = {}
    if net_factory is PauliNet:
        mole = pyscf.gto.M(atom=mol.as_pyscf(), unit='bohr', basis='6-311g', cart=True)
        basis = GTOBasis.from_pyscf(mole)
        args += (basis,)
        kwargs.update({'omni_factory': OmniNet})
    return net_factory(*args, **kwargs)


def test_batching(wf, rs):
    assert_alltrue_named(
        (name, torch.allclose(wf(rs[:2])[i], wf(rs)[i][:2], atol=0))
        for i, name in enumerate(['log(abs(psi))', 'sign(psi)'])
    )


def test_antisymmetry(wf, rs):
    assert_alltrue_named(
        (name, torch.allclose(wf(rs[:, [0, 2, 1]])[i], (-1) ** i * wf(rs)[i]))
        for i, name in enumerate(['log(abs(psi))', 'sign(psi)'])
    )


def test_antisymmetry_trained(wf, rs):
    sampler = LangevinSampler(wf, torch.rand_like(rs), tau=0.1)
    fit_wf(
        wf, LossEnergy(), torch.optim.Adam(wf.parameters(), lr=1e-2), sampler, range(10)
    )
    assert_alltrue_named(
        (name, torch.allclose(wf(rs[:, [0, 2, 1]])[i], (-1) ** i * wf(rs)[i]))
        for i, name in enumerate(['log(abs(psi))', 'sign(psi)'])
    )


def test_backprop(wf, rs):
    wf(rs)[0].sum().backward()
    assert_alltrue_named(
        (name, param.grad is not None) for name, param in wf.named_parameters()
    )
    assert_alltrue_named(
        (name, (param.grad.sum().abs().item() > 0 or name == 'mo.cusp_corr.shifts'))
        for name, param in wf.named_parameters()
    )
    # mo.cusp_corr.shifts is excluded, as gradients occasionally vanish


def test_grad(wf, rs):
    rs.requires_grad_()
    wf(rs)[0].sum().backward()
    assert rs.grad.sum().abs().item() > 0


def test_loc_ene_backprop(wf, rs):
    rs.requires_grad_()
    Es_loc, _, _ = local_energy(rs, wf, create_graph=True)
    Es_loc.sum().backward()
    assert_alltrue_named(
        (name, (param.grad.sum().abs().item() > 0 or name == 'mo.cusp_corr.shifts'))
        for name, param in wf.named_parameters()
    )
    # mo.cusp_corr.shifts is excluded, as gradients occasionally vanish
