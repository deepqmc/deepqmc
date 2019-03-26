from .physics import local_energy


def loss_local_energy(Es_loc, psis, E_ref=None, correlated_sampling=True):
    if correlated_sampling:
        ws = psis ** 2 / psis.detach() ** 2
        w_mean = ws.mean()
    else:
        ws, w_mean = 1.0, 1.0
    E0 = E_ref if E_ref is not None else (Es_loc * ws).mean() / w_mean
    return (ws * (Es_loc - E0) ** 2).mean() / w_mean


def fit_wfnet(wfnet, loss_func, opt, sampler, writer=None):
    for step, rs in enumerate(sampler):
        Es_loc, psis = local_energy(rs, wfnet, wfnet.geom, create_graph=True)
        loss = loss_func(Es_loc, psis)
        if writer:
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('E_loc/mean', Es_loc.mean(), step)
            writer.add_scalar('E_loc/var', Es_loc.var(), step)
            writer.add_scalar('param/ion_pot', wfnet.ion_pot, step)
        loss.backward()
        opt.step()
        opt.zero_grad()
