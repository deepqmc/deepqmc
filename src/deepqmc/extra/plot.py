from functools import partial

import torch

from .analysis import get_flat_mesh

__all__ = ()


def plot_func(
    func,
    bounds,
    density=0.02,
    x_line=False,
    is_torch=True,
    device=None,
    double=False,
    ax=None,
    **kwargs,
):
    if not ax:
        import matplotlib.pyplot as plt

        ax = plt.gca()
    n_pts = int((bounds[1] - bounds[0]) / density)
    x = torch.linspace(bounds[0], bounds[1], n_pts)
    if x_line:
        x = torch.cat([x[:, None], x.new_zeros((n_pts, 2))], dim=1)
    if not is_torch:
        x = x.numpy()
    else:
        if device:
            x = x.to(device)
        if double:
            x = x.double()
    y = func(x)
    if is_torch:
        x = x.cpu().numpy()
        y = y.detach().cpu().numpy()
    if x_line:
        x = x[:, 0]
    return ax.plot(x, y, **kwargs)


def plot_func_2d(
    func,
    bounds,
    density=0.02,
    xy_plane=False,
    device=None,
    ax=None,
    plot='contour',
    **kwargs,
):
    if not ax:
        import matplotlib.pyplot as plt

        ax = plt.gca()
    ns_pts = [int((bs[1] - bs[0]) / density) for bs in bounds]
    xy, x_y = get_flat_mesh(bounds, ns_pts, device=device)
    if xy_plane:
        xy = torch.cat([xy, xy.new_zeros(len(xy), 1)], dim=1)
    res = getattr(ax, plot)(
        *(z.cpu().numpy() for z in x_y),
        func(xy).detach().view(len(x_y[0]), -1).cpu().numpy().T,
        **kwargs,
    )
    ax.set_aspect(1)
    return res


plot_func_x = partial(plot_func, x_line=True)
plot_func_xy = partial(plot_func_2d, xy_plane=True)
