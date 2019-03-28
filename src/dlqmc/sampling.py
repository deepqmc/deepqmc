import numpy as np
import pandas as pd
import torch
from tqdm import trange

from .physics import quantum_force
from .utils import assign_where


def langevin_monte_carlo(wf, r, n_steps, tau, range=range):
    n_walker = r.shape[0]
    samples = r.new_empty(n_walker, n_steps, 3)
    force, psi = quantum_force(r, wf)
    n_accepted = 0
    for i in range(n_steps):
        r_new = r + force * tau + torch.randn_like(r) * np.sqrt(tau)
        force_new, psi_new = quantum_force(r_new, wf)
        log_G_ratio = (
            (force + force_new) * ((r - r_new) + tau / 2 * (force - force_new))
        ).sum(dim=-1)
        P_acc = torch.exp(log_G_ratio) * psi_new ** 2 / psi ** 2
        accepted = P_acc > torch.rand_like(P_acc)
        n_accepted += accepted.type(torch.int).sum().item()
        samples[:, i] = r
        assign_where((r, psi, force), (r_new, psi_new, force_new), accepted)
    info = {'acceptance': n_accepted / (n_walker * n_steps)}
    return samples, info
    
def dynamics(dist, pos, stepsize, steps):

    pos = pos.detach().clone()
    pos.requires_grad = True
    vel = torch.randn(pos.shape)
    v_te2 = (
        vel
        - stepsize
        * torch.autograd.grad(
            dist(pos),
            pos,
            create_graph=True,
            retain_graph=True,
            grad_outputs=torch.ones(pos.shape[0]),
        )[0]
        / 2
    )
    p_te = pos + stepsize * v_te2
    for _ in range(1, steps):
        v_te2 = (
            v_te2
            - stepsize
            * torch.autograd.grad(
                dist(p_te),
                p_te,
                create_graph=True,
                retain_graph=True,
                grad_outputs=torch.ones(pos.shape[0]),
            )[0]
        )
        p_te = p_te + stepsize * v_te2

    v_te = (
        v_te2
        - stepsize / 2
        * torch.autograd.grad(
            dist(p_te),
            p_te,
            create_graph=True,
            retain_graph=True,
            grad_outputs=torch.ones(pos.shape[0]),
        )[0])

    return p_te, v_te, vel


def hmc(
    dist,
    stepsize,
    dysteps,
    n_walker,
    steps,
    dim,
    startfactor=1,
    presteps=200,
):

    acc = 0
    samples = torch.zeros(steps, n_walker, dim)
    walker = torch.randn(n_walker, dim) * startfactor
    distwalker = dist(walker).detach().numpy()

    for i in trange(steps + presteps):

        if i >= presteps:
            samples[i - presteps] = walker

        trial, v_trial, v_0 = dynamics((lambda x: -torch.log(dist(x))), walker, stepsize, dysteps)
        disttrial = dist(trial).detach().numpy()
        ratio = torch.from_numpy(disttrial / distwalker) * (
            torch.exp(
                -0.5
                * (torch.sum(v_trial ** 2, dim=-1) - torch.sum(v_0 ** 2, dim=-1))
            )
        )
        R = torch.rand(n_walker)
        smaller = (ratio < R).type(torch.LongTensor)
        larger = torch.abs(smaller - 1)
        ind = torch.nonzero(larger).flatten()
        walker[ind] = trial[ind]
        distwalker[ind] = disttrial[ind]
        if i >= presteps:
            acc += torch.sum(larger).item()

    print('Acceptanceratio: ' + str(np.round(acc / (n_walker * steps) * 100, 2)) + '%')
    return samples


def metropolis(
    distribution,
    startpoint,
    stepsize,
    steps,
    dim,
    n_walker,
    startfactor=0.2,
    presteps=0,
    interval=None,
    T=0.2,
):

    samples = torch.zeros(steps, n_walker, len(startpoint))
    ratios = np.zeros((steps, n_walker))
    walker = torch.randn(n_walker, dim) * startfactor
    distwalker = distribution(walker)
    for i in trange(presteps + steps):
        if i > (presteps - 1):
            samples[i - presteps] = walker
        pro = (torch.rand(walker.shape) - 0.5) * stepsize
        trial = walker + pro
        disttrial = distribution(trial)
        
        if interval is not None:
            inint = torch.tensor(
                all(torch.tensor(interval[0]).type(torch.FloatTensor) < trial[0])
                and all(torch.tensor(interval[1]).type(torch.FloatTensor) > trial[0])
            ).type(torch.FloatTensor)
            disttrial = disttrial * inint

        ratio = np.exp((disttrial.detach().numpy() - distwalker.detach().numpy()) / T)
        ratios[i - presteps] = ratio
        smaller_n = (ratio < np.random.uniform(0, 1, n_walker)).astype(float)
        smaller = torch.from_numpy(smaller_n).type(torch.FloatTensor)
        larger = torch.abs(smaller - 1)
        walker = trial * larger[:, None] + walker * smaller[:, None]

    return samples

def samples_from(sampler, steps, *, n_discard=0):
    samples, infos = zip(*(step for step, i in zip(sampler, steps) if i >= n_discard))
    return torch.stack(samples, dim=1), pd.DataFrame(infos)


def langevin_monte_carlo(wf, rs, *, tau):
    forces, psis = quantum_force(rs, wf)
    while True:
        rs_new = rs + forces * tau + torch.randn_like(rs) * np.sqrt(tau)
        forces_new, psis_new = quantum_force(rs_new, wf)
        log_G_ratios = (
            (forces + forces_new) * ((rs - rs_new) + tau / 2 * (forces - forces_new))
        ).sum(dim=(-1, -2))
        Ps_acc = torch.exp(log_G_ratios) * psis_new ** 2 / psis ** 2
        accepted = Ps_acc > torch.rand_like(Ps_acc)
        info = {'acceptance': accepted.type(torch.int).sum().item() / rs.shape[0]}
        yield rs.clone(), info
        assign_where((rs, psis, forces), (rs_new, psis_new, forces_new), accepted)

