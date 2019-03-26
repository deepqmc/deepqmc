import numpy as np
import torch

from .physics import quantum_force
from .utils import assign_where


def langevin_monte_carlo(wf, rs, n_steps, tau, range=range):
    n_walker = rs.shape[0]
    samples = rs.new_empty(n_walker, n_steps, 3)
    forces, psis = quantum_force(rs, wf)
    n_accepted = 0
    for i in range(n_steps):
        rs_new = rs + forces * tau + torch.randn_like(rs) * np.sqrt(tau)
        forces_new, psis_new = quantum_force(rs_new, wf)
        log_G_ratios = (
            (forces + forces_new) * ((rs - rs_new) + tau / 2 * (forces - forces_new))
        ).sum(dim=-1)
        Ps_acc = torch.exp(log_G_ratios) * psis_new ** 2 / psis ** 2
        accepted = Ps_acc > torch.rand_like(Ps_acc)
        n_accepted += accepted.type(torch.int).sum().item()
        samples[:, i] = rs
        assign_where((rs, psis, forces), (rs_new, psis_new, forces_new), accepted)
    info = {'acceptance': n_accepted / (n_walker * n_steps)}
    return samples, info
