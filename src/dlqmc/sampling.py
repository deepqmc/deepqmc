from functools import partial

import numpy as np
import pandas as pd
import torch

from . import torchext
from .physics import quantum_force
from .utils import assign_where


def dynamics(wf, pos, stepsize, steps, tau, cutoff):
    qforce = partial(quantum_force, clamp=cutoff / tau)
    pos = pos.detach().clone()
    pos.requires_grad = True
    vel = torch.randn(pos.shape, device=pos.device)
    forces, psis = qforce(pos, wf)
    v_te2 = vel + stepsize * forces
    p_te = pos + stepsize * v_te2
    for _ in range(1, steps):
        forces, psis = qforce(p_te, wf)
        v_te2 = v_te2 + stepsize * 2 * forces
        p_te = p_te + stepsize * v_te2
    forces, psis = qforce(p_te, wf)
    v_te = v_te2 + stepsize * forces
    return p_te, v_te, vel


def hmc(wf, rs, *, dysteps, stepsize, tau, cutoff=1.0):
    while True:
        rs_new, v, v_0 = dynamics(wf, rs, stepsize, dysteps, tau, cutoff)
        psis = wf(rs)
        Ps_acc = (
            wf(rs_new) ** 2
            / psis ** 2
            * (
                torch.exp(
                    -0.5
                    * (
                        torch.sum(v ** 2, dim=[-1, -2])
                        - torch.sum(v_0 ** 2, dim=[-1, -2])
                    )
                )
            )
        )
        accepted = Ps_acc > torch.rand_like(Ps_acc)
        info = {'acceptance': accepted.type(torch.int).sum().item() / rs.shape[0]}
        yield rs.clone(), psis.clone(), info
        assign_where((rs,), (rs_new,), accepted)


def metropolis(wf, rs, *, stepsize):
    while True:
        rs_new = torch.randn_like(rs) * stepsize
        psis = wf(rs)
        Ps_acc = wf(rs_new) ** 2 / psis ** 2
        accepted = Ps_acc > torch.rand_like(Ps_acc)
        info = {'acceptance': accepted.type(torch.int).sum().item() / rs.shape[0]}
        yield rs.clone(), psis.clone(), info
        assign_where((rs,), (rs_new,), accepted)


def samples_from(sampler, steps, *, n_discard=0, n_decorrelate=0):
    rs, psis, infos = zip(*(step for step, i in zip(sampler, steps) if (i >= n_discard)&(i%(n_decorrelate+1)==0)))
    return torch.stack(rs, dim=1), torch.stack(psis, dim=1), pd.DataFrame(infos)


def langevin_monte_carlo(wf, rs, *, tau, cutoff=1.0):
    qforce = partial(quantum_force, clamp=cutoff / tau)
    forces, psis = qforce(rs, wf)
    while True:
        rs_new = rs + forces * tau + torch.randn_like(rs) * np.sqrt(tau)
        try:
            forces_new, psis_new = qforce(rs_new, wf)
        except torchext.LUFactError as e:
            e.info['rs'] = rs[e.info['idxs']]
            raise
        log_G_ratios = (
            (forces + forces_new) * ((rs - rs_new) + tau / 2 * (forces - forces_new))
        ).sum(dim=(-1, -2))
        Ps_acc = torch.exp(log_G_ratios) * psis_new ** 2 / psis ** 2
        accepted = Ps_acc > torch.rand_like(Ps_acc)
        
        
def take(a,n):
    l=[]
    while n>len(a):
        n=n-len(a)
        l.append(np.random.choice(a,len(a),replace=False))
    l.append(np.random.choice(a,n,replace=False))
    return np.random.permutation(np.concatenate(l))

def sample_start(geom,n_walker,n_electrons,var=1,cuda=True):
    ind = np.array([take(np.repeat(np.arange(0,len(hn._charges)),(hn._charges.numpy().astype(int))),n_electrons)for i in range(n_walker)])
    pos=torch.randn(n_walker,n_electrons,3)*var+torch.from_numpy(hn.coords[None,:,:].numpy().take(ind,axis=1)).view(-1,n_electrons,3)
    if cuda:
        return pos.cuda()
    else:
        return pos
        info = {'acceptance': accepted.type(torch.int).sum().item() / rs.shape[0]}
        yield rs.clone(), psis.clone(), info
        assign_where((rs, psis, forces), (rs_new, psis_new, forces_new), accepted)
