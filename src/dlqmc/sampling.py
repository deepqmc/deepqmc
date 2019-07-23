from functools import partial

import numpy as np
import pandas as pd
import torch

from . import torchext
from .physics import quantum_force
from .utils import assign_where

def samples_from(sampler, steps, *, n_discard=0, n_decorrelate=0):
    rs, psis, infos = zip(
        *(
            step
            for i, step in zip(steps, sampler)
            if i >= n_discard and i % (n_decorrelate + 1) == 0
        )
    )
    return torch.stack(rs, dim=1), torch.stack(psis, dim=1), pd.DataFrame(infos)

class LangevinSampler():

    def __init__(self, wf, rs, tau, wf_threshold=None, cutoff=1.0):
        self.wf, self.rs, self.tau  = wf,rs,tau
        self.cutoff = cutoff
        self.wf_threshold = wf_threshold
        self._lifetime = rs.new_zeros(len(rs), dtype=torch.long)
        self.recompute_forces()
        
    def __iter__(self):
        return (self)
        
    def _walker_step(self):
        return self.rs + self.forces * self.tau + torch.randn_like(self.rs) * np.sqrt(self.tau)
        
    def qforce(self,rs):
        try:
            return quantum_force(rs, self.wf, clamp=self.cutoff / self.tau)
        except torchext.LUFactError as e:
            e.info['rs'] = rs[e.info['idxs']]
            raise
        
    def __next__(self):
        rs_new = self._walker_step()
        forces_new, psis_new = self.qforce(rs_new)
        log_G_ratios = ((self.forces + forces_new) * ((self.rs - rs_new) + self.tau / 2 * (self.forces - forces_new))).sum(dim=(-1, -2))
        Ps_acc = torch.exp(log_G_ratios) * psis_new ** 2 / self.psis ** 2
        accepted = Ps_acc > torch.rand_like(Ps_acc) 
        accepted = accepted & (torch.abs(psis_new)>self.wf_threshold) if self.wf_threshold is not None else accepted
        self._lifetime[accepted] = 0
        self._lifetime[~accepted] += 1
        info = {
            'acceptance': accepted.type(torch.int).sum().item() / self.rs.shape[0],
            'lifetime': self._lifetime.cpu().numpy(),
        }
        assign_where((self.rs, self.psis, self.forces), (rs_new, psis_new, forces_new), accepted)
        return self.rs.clone(), self.psis.clone(), info
    
    def __repr__(self):
        return f'Object of class Sampler.\nn_walker = {self.rs.shape[0]}, n_electrons = {self.rs.shape[1]}, tau = {self.tau}'
   
    def __str__(self):
        return f'Object of class Sampler.\nn_walker = {self.rs.shape[0]}, n_electrons = {self.rs.shape[1]}, tau = {self.tau}'
   
    def propagate_all(self):
        self.rs = self._walker_step()
        self.recompute_forces()
    
    def recompute_forces(self):
        self.forces, self.psis = self.qforce(self.rs)

def rand_from_mf(mf, bs, charge_std=0.25, elec_std=1.0, idxs=None):
    mol = mf.mol
    n_atoms = mol.natm
    charges = mol.atom_charges()
    n_electrons = charges.sum() - mol.charge
    while idxs is None:
        cs = torch.tensor(charges - mf.pop(verbose=0)[1]).float()
        cs = cs + charge_std * torch.randn(bs, n_atoms)
        repeats = (cs / cs.sum(dim=-1)[:, None] * n_electrons).round().to(torch.long)
        try:
            idxs = torch.repeat_interleave(
            torch.arange(n_atoms).expand(bs, -1), repeats.flatten()
        ).view(bs, n_electrons)
        except RuntimeError: continue
    idxs = torch.stack([idxs[i,torch.randperm(idxs.shape[-1])] for i in range(bs)])

    centers = torch.tensor(mol.atom_coords()).float()[idxs]
    rs = centers + elec_std * torch.randn_like(centers)
    return rs
