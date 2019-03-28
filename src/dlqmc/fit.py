from itertools import cycle

from torch.utils.data import DataLoader

from .physics import local_energy
from .sampling import samples_from


def loss_local_energy(Es_loc, psis, E_ref=None, correlated_sampling=True):
    if correlated_sampling:
        ws = psis ** 2 / psis.detach() ** 2
        w_mean = ws.mean()
    else:
        ws, w_mean = 1.0, 1.0
    E0 = E_ref if E_ref is not None else (Es_loc * ws).mean() / w_mean
    return (ws * (Es_loc - E0) ** 2).mean() / w_mean


def fit_wfnet(wfnet, loss_func, opt, sample_gen, writer=None):
    for step, rs in enumerate(sample_gen):
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


def wfnet_fit_driver(
    sampler,
    *,
    samplings,
    n_epochs,
    n_sampling_steps,
    batch_size=10_000,
    n_discard=50,
    range_sampling=range,
    range_training=range,
):
    for _ in samplings:
        samples, _ = samples_from(
            sampler, range_sampling(n_sampling_steps), n_discard=n_discard
        )
        rs_dl = DataLoader(
            samples.flatten(end_dim=1), batch_size=batch_size, shuffle=True
        )
        n_steps = n_epochs * len(rs_dl)
        for _, rs in zip(range_training(n_steps), cycle(rs_dl)):
            yield rs
