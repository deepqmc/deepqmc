import numpy as np
import pandas as pd
import torch

from .physics import quantum_force
from .utils import assign_where


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
        ).sum(dim=-1)
        Ps_acc = torch.exp(log_G_ratios) * psis_new ** 2 / psis ** 2
        accepted = Ps_acc > torch.rand_like(Ps_acc)
        info = {'acceptance': accepted.type(torch.int).sum().item() / rs.shape[0]}
        yield rs.clone(), info
        assign_where((rs, psis, forces), (rs_new, psis_new, forces_new), accepted)
