import numpy as np
import pandas as pd
import torch

from . import torchext
from .physics import quantum_force
from .utils import assign_where


def dynamics(wf, pos, stepsize, steps):
    pos = pos.detach().clone()
    pos.requires_grad = True
    vel = torch.randn(pos.shape)
    forces, psis = quantum_force(pos, wf)
    v_te2 = vel + stepsize * forces
    p_te = pos + stepsize * v_te2
    for _ in range(1, steps):
        forces, psis = quantum_force(p_te, wf)
        v_te2 = v_te2 + stepsize * 2 * forces
        p_te = p_te + stepsize * v_te2
    forces, psis = quantum_force(p_te, wf)
    v_te = v_te2 + stepsize * forces
    return p_te, v_te, vel


def hmc(wf, rs, *, dysteps, stepsize):
    while True:
        rs_new, v, v_0 = dynamics(wf, rs, stepsize, dysteps)
        Ps_acc = (
            wf(rs_new) ** 2
            / wf(rs) ** 2
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
        yield rs.clone(), info
        assign_where((rs,), (rs_new,), accepted)


def metropolis(wf, rs, *, stepsize):
    while True:
        rs_new = torch.randn_like(rs) * stepsize
        Ps_acc = wf(rs_new) ** 2 / wf(rs) ** 2
        accepted = Ps_acc > torch.rand_like(Ps_acc)
        info = {'acceptance': accepted.type(torch.int).sum().item() / rs.shape[0]}
        yield rs.clone(), info
        assign_where((rs,), (rs_new,), accepted)


def samples_from(sampler, steps, *, n_discard=0):
    samples, infos = zip(*(step for step, i in zip(sampler, steps) if i >= n_discard))
    return torch.stack(samples, dim=1), pd.DataFrame(infos)


def langevin_monte_carlo(wf, rs, cutoff=1.0, *, tau):
    def cutoff_force(rs, wf):
        forces, psis = quantum_force(rs, wf)
        max_force = torch.tensor(cutoff / tau)
        forces_norm = forces.norm(dim=-1)
        norm_factors = torch.min(forces_norm, max_force) / forces_norm
        return (forces * norm_factors[..., None], psis)

    forces, psis = cutoff_force(rs, wf)
    while True:
        rs_new = rs + forces * tau + torch.randn_like(rs) * np.sqrt(tau)
        try:
            forces_new, psis_new = cutoff_force(rs_new, wf)
        except torchext.LUFactError as e:
            e.info['rs'] = rs[e.info['idxs']]
            raise
        log_G_ratios = (
            (forces + forces_new) * ((rs - rs_new) + tau / 2 * (forces - forces_new))
        ).sum(dim=(-1, -2))
        Ps_acc = torch.exp(log_G_ratios) * psis_new ** 2 / psis ** 2
        accepted = Ps_acc > torch.rand_like(Ps_acc)
        info = {'acceptance': accepted.type(torch.int).sum().item() / rs.shape[0]}
        yield rs.clone(), info
        assign_where((rs, psis, forces), (rs_new, psis_new, forces_new), accepted)
