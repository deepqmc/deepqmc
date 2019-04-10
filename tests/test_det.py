import pytest
import torch

from dlqmc import torchext


@pytest.fixture
def xs():
    return torch.randn(10, 4, 4).double().requires_grad_()


def test_1st_deriv(xs):
    assert torch.autograd.gradcheck(torchext.bdet, xs)


def test_2nd_deriv(xs):
    assert torch.autograd.gradgradcheck(torchext.bdet, xs)


def test_3rd_deriv(xs):
    def func(xs):
        ys = torchext.bdet(xs)
        dys, = torch.autograd.grad(ys, xs, torch.ones_like(ys), create_graph=True)
        ddys, = torch.autograd.grad(dys, xs, torch.ones_like(xs), create_graph=True)
        return (ddys ** 2).sum()

    assert torch.autograd.gradcheck(func, xs)
