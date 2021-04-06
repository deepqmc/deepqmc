import torch

__all__ = ()


def jacobian(out, inp):
    jac = out.new_zeros(out.numel(), inp.numel())
    grad_out = torch.empty_like(out)
    for i in range(out.numel()):
        grad_out.zero_().flatten()[i] = 1
        jac[i, :] = torch.autograd.grad(out, inp, grad_out, retain_graph=True)[
            0
        ].flatten()
    return jac


def numjacobian(f, inp, degree=5, delta=1e-3, return_diffs=False):
    idx0 = (degree - 1) // 2
    out = f(inp)
    jac = out.new_zeros(out.numel(), inp.numel(), degree)
    jac[:, :, idx0] = out.flatten()[:, None]
    inp_delta = torch.empty_like(inp)
    for i in range(inp.numel()):
        for step in range(-idx0, idx0 + 1):
            if step == 0:
                continue
            inp_delta.copy_(inp).flatten()[i] += step * delta
            jac[:, i, idx0 + step] = f(inp_delta).flatten()
    if return_diffs:
        return jac
    coeffs = {
        3: [-1 / 2, 0, 1 / 2],
        5: [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12],
        7: [-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60],
    }[degree]
    jac = jac @ jac.new_tensor(coeffs) / delta
    return jac


def laplacian_fd(
    xs, f, create_graph=False, keep_graph=None, return_grad=False, eps=0.3e-2
):
    assert not create_graph
    xs = xs.detach().requires_grad_()
    ys = f(xs)
    (ys_g, *other) = ys if isinstance(ys, tuple) else (ys, ())
    ones = torch.ones_like(ys_g)
    (dy_dxs,) = torch.autograd.grad(ys_g, xs, ones, retain_graph=True)
    dev = xs.device
    n, N = xs.shape[:2]
    dim = 3 * N
    xs = torch.cat(
        [
            xs.unsqueeze(dim=1) + eps * torch.eye(dim, device=dev).view(-1, N, 3),
            xs.unsqueeze(dim=1) - eps * torch.eye(dim, device=dev).view(-1, N, 3),
        ],
        dim=1,
    )
    with torch.no_grad():
        lap_ys = f(xs.flatten(end_dim=1))
    (lap_ys, *other) = lap_ys if isinstance(lap_ys, tuple) else (lap_ys, ())
    lap_ys = lap_ys.view(n, -1)
    other = [o.view(n, -1)[:, -1] for o in other]
    lap_ys = (
        (lap_ys[:, :dim] + lap_ys[:, dim:]).sum(dim=-1) - dim * 2 * ys_g.detach()
    ) / eps ** 2
    if not (create_graph if keep_graph is None else keep_graph):
        ys = (ys_g.detach(), *other) if isinstance(ys, tuple) else ys.detach()
    result = lap_ys, ys
    if return_grad:
        result += (dy_dxs,)
    return result


def laplacian_stochastic(xs, f, create_graph=False, keep_graph=None, return_grad=False):
    xis = [xi.requires_grad_() for xi in xs.flatten(start_dim=1).t()]
    xs_flat = torch.stack(xis, dim=1)
    ys = f(xs_flat.view_as(xs))
    (ys_g, *other) = ys if isinstance(ys, tuple) else (ys, ())
    v = torch.randn_like(xs_flat)
    (dy_dxs,) = torch.autograd.grad(ys_g.sum(), xs_flat, create_graph=True)
    (d2y_dx2s,) = torch.autograd.grad(
        (dy_dxs * v).sum(), xs_flat, create_graph=create_graph, retain_graph=True
    )
    lap_ys = (d2y_dx2s * v).sum(dim=-1)
    if not (create_graph if keep_graph is None else keep_graph):
        ys = (ys_g.detach(), *other) if isinstance(ys, tuple) else ys.detach()
    result = lap_ys, ys
    if return_grad:
        result += (dy_dxs.detach().view_as(xs),)
    return result
