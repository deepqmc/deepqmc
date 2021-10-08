from functools import partial

import torch

from deepqmc.sampling import rand_from_mol

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


def rs_mesh(n_electrons, dims, bounds, steps, cut=None):
    flat_mesh, _ = get_flat_mesh(bounds, steps)
    cut = cut.flatten() if cut is not None else torch.randn(3 * n_electrons)
    rs = cut.repeat(len(flat_mesh), 1)
    rs[:, dims] = flat_mesh
    return rs.view(*steps, n_electrons, 3)


@torch.no_grad()
def extract_nodes(wf, mesh, dim):
    assert dim in [2, 3]
    steps = mesh.shape[:-2]
    psis, signs = map(lambda x: x.view(1, 1, *steps), wf(mesh.flatten(end_dim=-3)))
    if dim == 2:
        surf = torch.nn.functional.conv2d(signs, torch.ones(1, 1, 2, 2))
    elif dim == 3:
        surf = torch.nn.functional.conv3d(signs, torch.ones(1, 1, 2, 2, 2))
    return surf[0, 0], psis, signs


def plot_nodal_surface(
    wf,
    mol,
    dimensions,
    bounds,
    steps,
    cut=None,
    fig=None,
    plot_kwargs=None,
):
    from plotly.graph_objects import Isosurface
    from plotly.subplots import make_subplots

    assert len(dimensions) == 3
    bounds, steps = torch.tensor(bounds), torch.tensor(steps)
    bounds = bounds.repeat(3, 1) if len(bounds.shape) == 1 else bounds
    steps = steps.repeat(3) if not len(steps.shape) else steps
    cut = rand_from_mol(mol, 1)[0] if cut is None else cut
    fig = fig or make_subplots()
    plot_kwargs = {
        'isomin': 0,
        'isomax': 0,
        'colorscale': 'oranges',
        'surface_count': 1,
        'opacity': 0.4,
        **(plot_kwargs or {}),
    }
    n_el = int(sum(mol.charges) - mol.charge)
    mesh = rs_mesh(n_el, dimensions, bounds.tolist(), steps.tolist(), cut)
    surf, _, _ = extract_nodes(wf, mesh, 3)
    x, y, z = get_flat_mesh(bounds.tolist(), (steps - 1).tolist())[0].t()
    fig.add_trace(
        Isosurface(x=x, y=y, z=z, value=surf.flatten(), showscale=False, **plot_kwargs)
    )
    if all(torch.tensor(dimensions) // 3 == dimensions[0] // 3):
        for i, ri in enumerate(mesh.flatten(end_dim=-3)[0]):
            if i not in torch.tensor(dimensions) // 3:
                x, y, z = ri.unsqueeze(-1).numpy()
                fig.add_scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode='markers',
                    marker={'size': 4},
                    name=f'electron #{i+1}',
                )
    for ri, zi in zip(mol.coords, mol.charges):
        x, y, z = ri.unsqueeze(-1).numpy()
        fig.add_scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker={'size': zi.item() + 4, 'color': [-1], 'colorscale': 'greys'},
            name=f'nucleus Z={zi}',
        )
    return fig
