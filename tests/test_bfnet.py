import pytest
import torch
from torch.testing import assert_allclose

from dlqmc.geom import Geometry, angstrom
from dlqmc.nn import BFNet


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
