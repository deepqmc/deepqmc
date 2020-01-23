import torch

__all__ = ()


def grad(xs, f, create_graph=False):
    xs = xs if xs.requires_grad else xs.detach().requires_grad_()
    ys = f(xs)
    (ys_g, *other) = ys if isinstance(ys, tuple) else (ys, ())
    (grad_ys,) = torch.autograd.grad(
        ys_g, xs, torch.ones_like(ys_g), create_graph=create_graph
    )
    if not create_graph:
        ys = (ys_g.detach(), *other) if isinstance(ys, tuple) else ys.detach()
    return grad_ys, ys


def laplacian(xs, f, create_graph=False, keep_graph=None, return_grad=False):
    xis = [xi.requires_grad_() for xi in xs.flatten(start_dim=1).t()]
    xs_flat = torch.stack(xis, dim=1)
    ys = f(xs_flat.view_as(xs))
    (ys_g, *other) = ys if isinstance(ys, tuple) else (ys, ())
    ones = torch.ones_like(ys_g)
    (dy_dxs,) = torch.autograd.grad(ys_g, xs_flat, ones, create_graph=True)
    lap_ys = sum(
        torch.autograd.grad(
            dy_dxi, xi, ones, retain_graph=True, create_graph=create_graph
        )[0]
        for xi, dy_dxi in zip(xis, (dy_dxs[..., i] for i in range(len(xis))))
    )
    if not (create_graph if keep_graph is None else keep_graph):
        ys = (ys_g.detach(), *other) if isinstance(ys, tuple) else ys.detach()
    result = lap_ys, ys
    if return_grad:
        result += (dy_dxs.detach().view_as(xs),)
    return result
