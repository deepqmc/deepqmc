import operator
from functools import reduce

import torch


def grad(xs, model, create_graph=False):
    xs = xs if xs.requires_grad else xs.detach().requires_grad_()
    ys = model(xs)
    grad_ys, = torch.autograd.grad(
        ys, xs, grad_outputs=torch.ones_like(ys), create_graph=create_graph
    )
    if not create_graph:
        ys.detach_()
    return grad_ys, ys


def laplacian(xs, model, create_graph=False):
    xis = [xi.requires_grad_() for xi in xs.flatten(start_dim=1).t()]
    ys = model(torch.stack(xis, dim=1).view_as(xs))
    ones = torch.ones_like(ys)

    def d2y_dxi2_from_xi(xi):
        dy_dxi, = torch.autograd.grad(ys, xi, grad_outputs=ones, create_graph=True)
        d2y_dxi2, = torch.autograd.grad(
            dy_dxi, xi, grad_outputs=ones, retain_graph=True, create_graph=create_graph
        )
        return d2y_dxi2

    lap_ys = reduce(operator.add, map(d2y_dxi2_from_xi, xis))
    if not create_graph:
        ys.detach_()
    return lap_ys, ys
