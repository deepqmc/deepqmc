import matplotlib.pyplot as plt
import torch


def plot_func(x, f, plot=plt.plot, **kwargs):
    return plot(x, f(x), **kwargs)


def get_3d_cube_mesh(bounds, npts):
    edges = [torch.linspace(*b, n) for b, n in zip(bounds, npts)]
    grids = torch.meshgrid(*edges)
    return torch.stack(grids).view(3, -1).t()


def assign_where(xs, ys, where):
    for x, y in zip(xs, ys):
        x[where] = y[where]
