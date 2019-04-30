from contextlib import contextmanager
from functools import partial, wraps

import matplotlib.pyplot as plt
import numpy as np
import torch


def triu_flat(x):
    i, j = np.triu_indices(x.shape[1], k=1)
    return x[:, i, j]


def get_flat_mesh(bounds, npts, device=None):
    edges = [torch.linspace(*b, n, device=device) for b, n in zip(bounds, npts)]
    grids = torch.meshgrid(*edges)
    return torch.stack(grids).flatten(start_dim=1).t(), edges


def plot_func(
    func, bounds, density=0.02, x_line=False, is_torch=True, device=None, **kwargs
):
    n_pts = int((bounds[1] - bounds[0]) / density)
    if x_line:
        x = torch.linspace(bounds[0], bounds[1], n_pts)
        x = torch.cat([x[:, None], x.new_zeros((n_pts, 2))], dim=1)
    else:
        x = torch.linspace(bounds[0], bounds[1], n_pts)
    if not is_torch:
        x = x.numpy()
    elif device:
        x = x.to(device)
    y = func(x)
    if is_torch:
        x = x.cpu().numpy()
        y = y.detach().cpu().numpy()
    if x_line:
        x = x[:, 0]
    return plt.plot(x, y, **kwargs)


plot_func_x = partial(plot_func, x_line=True)


def plot_func_xy(func, bounds, density=0.02, device=None):
    ns_pts = [int((bs[1] - bs[0]) / density) for bs in bounds]
    xy_plane, xy_edges = get_flat_mesh(bounds, ns_pts, device=device)
    xy_plane = torch.cat([xy_plane, xy_plane.new_zeros(len(xy_plane), 1)], dim=1)
    res = plt.contour(
        *(edge.cpu().numpy() for edge in xy_edges),
        func(xy_plane).detach().view(len(xy_edges[0]), -1).cpu().numpy().T,
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


class InfoException(Exception):
    def __init__(self, info=None):
        self.info = info or {}
        super().__init__(self.info)


def nondiag(A, k=None):
    A = A.copy()
    np.fill_diagonal(A, 0)
    return A


def dctsel(dct, keys):
    if isinstance(keys, str):
        keys = keys.split()
    return {k: dct[k] for k in keys if k in dct}


class DebugContainer(dict):
    def __init__(self):
        super().__init__()
        self._levels = []

    @contextmanager
    def cd(self, label):
        self._levels.append(label)
        try:
            yield
        finally:
            assert label == self._levels.pop()

    def __setitem__(self, key, val):
        super().__setitem__('.'.join([*self._levels, str(key)]), val)


class _NullDebug(DebugContainer):
    def __setitem__(self, key, val):
        pass


NULL_DEBUG = _NullDebug()


def debugged(func, label):
    @wraps(func)
    def wrapped(*args, **kwargs):
        debug = DebugContainer()
        func(*args, **kwargs, debug=debug)
        return debug[label]

    return wrapped


class Debuggable:
    def debug(self, label, *args, **kwargs):
        return debugged(self, label)(*args, **kwargs)


def batch_eval(func, batches, *args, **kwargs):
    return torch.cat([func(batch, *args, **kwargs) for batch in batches])


def batch_eval_tuple(func, batches, *args, **kwargs):
    results = list(zip(*(func(batch, *args, **kwargs) for batch in batches)))
    return tuple(torch.cat(result) for result in results)
