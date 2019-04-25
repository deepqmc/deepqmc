import pytest
import torch
from torch.testing import assert_allclose

from dlqmc.geom import Geometry, angstrom
from dlqmc.nn import BFNet
from dlqmc.physics import local_energy


def assert_alltrue_named(items):
    dct = dict(items)
    assert dct == {k: True for k in dct}


@pytest.fixture
def rs():
    return torch.randn(5, 3, 3)


@pytest.fixture
def h2_mol():
    return Geometry([[0, 0, 0], [0.742 * angstrom, 0, 0]], [1, 1])


@pytest.fixture
def bfnet(h2_mol):
    return BFNet(h2_mol, 3, 0)


def test_batching(bfnet, rs):
    assert_allclose(bfnet(rs[:2]), bfnet(rs)[:2])


def test_antisymmetry(bfnet, rs):
    assert_allclose(bfnet(rs[:, [0, 2, 1]]), -bfnet(rs))


def test_backprop(bfnet, rs):
    bfnet(rs).sum().backward()
    assert_alltrue_named(
        (name, param.grad is not None) for name, param in bfnet.named_parameters()
    )
    assert_alltrue_named(
        (name, param.grad.sum().abs().item() > 0)
        for name, param in bfnet.named_parameters()
    )


def test_grad(bfnet, rs):
    rs.requires_grad_()
    bfnet(rs).sum().backward()
    assert rs.grad.sum().abs().item() > 0


def test_loc_ene_backprop(bfnet, rs):
    rs.requires_grad_()
    Es_loc, _ = local_energy(rs, bfnet, bfnet.geom, create_graph=True)
    Es_loc.sum().backward()
    assert_alltrue_named(
        (name, param.grad.sum().abs().item() > 0)
        for name, param in bfnet.named_parameters()
    )
