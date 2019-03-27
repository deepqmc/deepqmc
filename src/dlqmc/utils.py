import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def get_flat_mesh(bounds, npts):
    edges = [torch.linspace(*b, n) for b, n in zip(bounds, npts)]
    grids = torch.meshgrid(*edges)
    return torch.stack(grids).flatten(start_dim=1).t(), edges


def plot_func(func, bounds, density=0.02, is_torch=True):
    n_pts = int((bounds[1] - bounds[0]) / density)
    x = (torch if is_torch else np).linspace(bounds[0], bounds[1], n_pts)
    y = func(x)
    if is_torch:
        x = x.numpy()
        y = y.detach().numpy()
    return plt.plot(x, y)


def plot_func_x(func, bounds, density=0.02, is_torch=True, device=None, **kwargs):
    n_pts = int((bounds[1] - bounds[0]) / density)
    x_line = torch.linspace(bounds[0], bounds[1], n_pts)
    x_line = torch.cat([x_line[:, None], x_line.new_zeros((n_pts, 2))], dim=1)
    if not is_torch:
        x_line = x_line.numpy()
    elif device:
        x_line = x_line.to(device)
    y = func(x_line)
    if is_torch:
        x_line = x_line.cpu().numpy()
        y = y.detach().cpu().numpy()
    return plt.plot(x_line[:, 0], y, **kwargs)


def plot_func_xy(func, bounds, density=0.02):
    ns_pts = [int((bs[1] - bs[0]) / density) for bs in bounds]
    xy_plane, xy_edges = get_flat_mesh(bounds, ns_pts)
    xy_plane = torch.cat([xy_plane, xy_plane.new_zeros(len(xy_plane), 1)], dim=1)
    res = plt.contour(
        *xy_edges, func(xy_plane).detach().view(len(xy_edges[0]), -1).numpy().T
    )
    plt.gca().set_aspect(1)
    return res


def integrate_on_mesh(func, bounds, density=0.02):
    ns_pts = [int((bs[1] - bs[0]) / density) for bs in bounds]
    vol = np.array([bs[1] - bs[0] for bs in bounds]).prod()
    mesh = get_flat_mesh(bounds, ns_pts)[0]
    return sum(func(x).sum() for x in mesh.chunk(100)) * (vol / mesh.shape[0])


def assign_where(xs, ys, where):
    for x, y in zip(xs, ys):
        x[where] = y[where]


def form_geom(coords, charges):
    coords = nn.Parameter(
        torch.as_tensor(coords, dtype=torch.float), requires_grad=False
    )
    charges = nn.Parameter(
        torch.as_tensor(charges, dtype=torch.float), requires_grad=False
    )
    return nn.ParameterDict({'coords': coords, 'charges': charges})


def as_pyscf_atom(geom):
    return [
        (str(int(charge.numpy())), coord.numpy())
        for charge, coord in zip(*geom.values())
    ]
