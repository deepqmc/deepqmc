import torch
import numpy as np

from ..geom import Geometry
from ..utils import NULL_DEBUG, triu_flat
from .anti import eval_slater
from .base import (
    BaseWFNet,
    DistanceBasis,
    ElectronicAsymptotic,
    pairwise_diffs,
    pairwise_distance,
)
from .gto import GTOBasis
from .molorb import MolecularOrbital

from dlqmc.nn.molorb import MolecularOrbital

from dlqmc.nn.molorb import MolecularOrbital


class SlaterJastrowNet(BaseWFNet):
    def __init__(
        self,
        geom,
        n_up,
        n_down,
        basis,
        mo_factory=None,
        jastrow_factory=None,
        dist_basis_dim=32,
        dist_basis_cutoff=10.0,
        cusp_correction=False,
        cusp_electrons=False,
        rc_scaling=1.0,
        activeorb = None,
        activeel = None,
        det_list = None,
        
    ):
        super().__init__()
        self.n_up, self.n_down = n_up, n_down
        
        if activeorb is None:
            self.activeorb = max(n_up,n_down)
        else:
            self.activeorb = activeorb
        
        if activeel is None:
            self.activeel = n_up+n_down
        else:
            self.activeel = activeel
            
        if det_list is None:
            self.det_list = [(1.,np.arange(n_up,dtype=float),np.arange(n_down,dtype=float))]
        else:
            self.det_list = det_list
        
        self.register_geom(geom)
        self.dist_basis = (
            DistanceBasis(dist_basis_dim, cutoff=dist_basis_cutoff, envelope='nocusp')
            if mo_factory or jastrow_factory
            else None
        )
        self.mo = MolecularOrbital(
            geom,
            basis,
            self.activeorb,
            net_factory=mo_factory,
            edge_dim=dist_basis_dim,
            cusp_correction=cusp_correction,
            rc_scaling=rc_scaling,
        )
        self.cusp_same, self.cusp_anti = (
            (ElectronicAsymptotic(cusp=cusp) for cusp in (0.25, 0.5))
            if cusp_electrons
            else (None, None)
        )
        self.jastrow = (
            jastrow_factory(len(geom), dist_basis_dim, n_up, n_down)
            if jastrow_factory
            else None
        )
        

    def init_from_pyscf(self, pyscf_input, **kwargs):
        self.mo.init_from_pyscf(pyscf_input, **kwargs)

        
    @classmethod
    def from_pyscf(cls, pyscf_input, cusp_correction=False, mo_cutoff=1e-3, **kwargs):
        
        n_up,n_down = pyscf_input.mol.nelec

        try:
            activeorb = pyscf_input.ncas
            activeel  = pyscf_input.nelecas
            det_list  = pyscf_input.fcisolver.large_ci(pyscf_input.ci, pyscf_input.ncas, pyscf_input.nelecas, tol=mo_cutoff, return_strs=False)

        except:
            activeorb = max(n_up,n_down)
            activeel  = pyscf_input.mol.nelectron
            det_list  = [(1.,np.arange(n_up,dtype=float),np.arange(n_down,dtype=float))]
            
        geom = Geometry(pyscf_input.mol.atom_coords().astype('float32'), pyscf_input.mol.atom_charges())
        basis = GTOBasis.from_pyscf(pyscf_input.mol)
        wf = cls(geom, n_up, n_down, basis,activeorb=activeorb,activeel=activeel,det_list=det_list, cusp_correction=cusp_correction)
        wf.init_from_pyscf(pyscf_input)
        return wf
    

    def forward(self, rs, debug=NULL_DEBUG):
        batch_dim, n_elec = rs.shape[:2]
        assert n_elec == self.n_up + self.n_down
        n_atoms = len(self.coords)
        diffs_nuc = pairwise_diffs(
            torch.cat([self.coords, rs.flatten(end_dim=1)]), self.coords
        )
        edges_nuc = (
            self.dist_basis(diffs_nuc[:, :, 3].sqrt())
            if self.jastrow or self.mo.net
            else None
        )
        xs = debug['mos'] = self.mo(diffs_nuc, edges_nuc, debug=debug).view(batch_dim,n_elec, self.activeorb)
        
        re_list = list(zip(*self.det_list))
        coeff = torch.tensor(re_list[0]).cuda()
        up_index = np.array(re_list[1])
        down_index = np.array(re_list[2])
        
        det_up = eval_slater(xs[:, : self.n_up, up_index].permute(0,2,1,3).flatten(end_dim=1)).view(batch_dim,-1)
        det_down = eval_slater(xs[:, self.n_up:, down_index].permute(0,2,1,3).flatten(end_dim=1)).view(batch_dim,-1)

        psi = torch.sum(coeff * det_up * det_down,dim=-1)
        
        if self.cusp_same or self.jastrow:
            dists_elec = pairwise_distance(rs, rs)
        if self.cusp_same:
            cusp_same = self.cusp_same(
                torch.cat(
                    [triu_flat(dists_elec[:, idxs, idxs]) for idxs in self.spin_slices],
                    dim=1,
                )
            )
            cusp_anti = self.cusp_anti(
                dists_elec[:, : self.n_up, self.n_up :].flatten(start_dim=1)
            )
            psi = psi * cusp_same * cusp_anti
        if self.jastrow:
            edges_nuc = edges_nuc[n_atoms:].view(batch_dim, n_elec, n_atoms, -1)
            edges = torch.cat([self.dist_basis(dists_elec), edges_nuc], dim=2)
            psi = psi * torch.exp(self.jastrow(edges))
        return psi
