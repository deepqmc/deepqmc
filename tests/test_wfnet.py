import pytest
import torch
from torch.testing import assert_allclose

from dlqmc.geom import geomdb
from dlqmc.nn import BFNet, HanNet
from dlqmc.physics import local_energy


def assert_alltrue_named(items):
    dct = dict(items)
    assert dct == {k: True for k in dct}


@pytest.fixture
def rs():
    return torch.randn(5, 3, 3)


@pytest.fixture
def geom():
    return geomdb['H2']


@pytest.fixture(params=[BFNet, HanNet])
def net_factory(request):
    return request.param


@pytest.fixture
def wfnet(net_factory, geom):
    return net_factory(geom, 3, 0)


def test_batching(wfnet, rs):
    assert_allclose(wfnet(rs[:2]), wfnet(rs)[:2])


def test_antisymmetry(wfnet, rs):
    assert_allclose(wfnet(rs[:, [0, 2, 1]]), -wfnet(rs))


def test_backprop(wfnet, rs):
    wfnet(rs).sum().backward()
    assert_alltrue_named(
        (name, param.grad is not None) for name, param in wfnet.named_parameters()
    )
    assert_alltrue_named(
        (name, param.grad.sum().abs().item() > 0)
        for name, param in wfnet.named_parameters()
    )


def test_grad(wfnet, rs):
    rs.requires_grad_()
    wfnet(rs).sum().backward()
    assert rs.grad.sum().abs().item() > 0


def test_loc_ene_backprop(wfnet, rs):
    rs.requires_grad_()
    Es_loc, _ = local_energy(rs, wfnet, create_graph=True)
    Es_loc.sum().backward()
    assert_alltrue_named(
        (name, param.grad.sum().abs().item() > 0)
        for name, param in wfnet.named_parameters()
    )
