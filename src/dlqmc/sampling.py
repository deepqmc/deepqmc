import numpy as np
import torch

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
