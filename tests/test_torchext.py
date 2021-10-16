import pytest
import torch
from torch.testing import assert_allclose

from deepqmc import torchext
from deepqmc.torchext import pow_int


class TestDet:
    @pytest.fixture
    def xs(self):
        return torch.randn(10, 4, 4).double().requires_grad_()

    def test_1st_deriv(self, xs):
        assert torch.autograd.gradcheck(torchext.bdet, xs)

    def test_2nd_deriv(self, xs):
        assert torch.autograd.gradgradcheck(torchext.bdet, xs)

    def test_3rd_deriv(self, xs):
        def func(xs):
            ys = torchext.bdet(xs)
            (dys,) = torch.autograd.grad(ys, xs, torch.ones_like(ys), create_graph=True)
            (ddys,) = torch.autograd.grad(
                dys, xs, torch.ones_like(xs), create_graph=True
            )
            return (ddys ** 2).sum()

        assert torch.autograd.gradcheck(func, xs)


class TestSloglindet:
    @pytest.fixture
    def xs(self):
        return (
            torch.randn(10).double().requires_grad_(),
            torch.randn(10, 3, 3).double().requires_grad_(),
            torch.randn(10, 2, 2).double().requires_grad_(),
        )

    def test_1st_deriv(self, xs):
        assert torch.autograd.gradcheck(torchext.sloglindet, xs)

    def test_2nd_deriv(self, xs):
        assert torch.autograd.gradgradcheck(torchext.sloglindet, xs)


class TestSloglindet1:
    @pytest.fixture
    def xs(self):
        return (
            torch.randn(10).double().requires_grad_(),
            torch.randn(10, 2, 2).double().requires_grad_(),
            torch.randn(10, 1, 1).double().requires_grad_(),
        )

    def test_1st_deriv(self, xs):
        assert torch.autograd.gradcheck(torchext.sloglindet, xs)

    def test_2nd_deriv(self, xs):
        assert torch.autograd.gradgradcheck(torchext.sloglindet, xs)


class TestSloglindet0:
    @pytest.fixture
    def xs(self):
        return (
            torch.randn(10).double().requires_grad_(),
            torch.randn(10, 2, 2).double().requires_grad_(),
            torch.randn(10, 0, 0).double().requires_grad_(),
        )

    def test_1st_deriv(self, xs):
        assert torch.autograd.gradcheck(torchext.sloglindet, xs)

    def test_2nd_deriv(self, xs):
        assert torch.autograd.gradgradcheck(torchext.sloglindet, xs)


def test_pow_int():
    xs = torch.randn(4, 3)
    exps = torch.tensor([(1, 2, 3), (0, 1, 2)])
    assert_allclose(pow_int(xs[:, None, :], exps), xs[:, None, :] ** exps.float())
