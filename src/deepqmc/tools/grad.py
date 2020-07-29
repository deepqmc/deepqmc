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
