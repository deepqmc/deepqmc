import torch
from torch import nn

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


class SlaterJastrowNet(BaseWFNet):
    def __init__(
        self,
        geom,
        n_up,
        n_down,
        basis,
        mo_factory=None,
        jastrow_factory=None,
        backflow_factory=None,
        omni_factory=None,
        dist_basis_dim=32,
        dist_basis_cutoff=10.0,
        cusp_correction=False,
        cusp_electrons=False,
        rc_scaling=1.0,
        configurations=None,
    ):
        super().__init__()
        self.n_up = n_up
        self.register_geom(geom)
        self.dist_basis = (
            DistanceBasis(dist_basis_dim, cutoff=dist_basis_cutoff, envelope='nocusp')
            if mo_factory or jastrow_factory or backflow_factory or omni_factory
            else None
        )
        if configurations is not None:
            assert configurations.shape[1] == n_up + n_down
            n_orbitals = configurations.max() + 1
            self.confs = configurations
            self.conf_coeff = nn.Linear(len(configurations), 1, bias=False)
        else:
            n_orbitals = max(n_up, n_down)
            self.confs = torch.cat(
                [torch.arange(n_up), torch.arange(n_down)]
            ).unsqueeze(dim=0)
            self.conf_coeff = nn.Identity()
        self.mo = MolecularOrbital(
            geom,
            basis,
            n_orbitals,
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
        self.backflow = (
            backflow_factory(len(geom), dist_basis_dim, n_up, n_down, n_orbitals)
            if backflow_factory
            else None
        )
        if omni_factory:
            assert not backflow_factory and not jastrow_factory
            self.omni = omni_factory(
                len(geom), dist_basis_dim, n_up, n_down, n_orbitals
            )
            self.backflow = self.omni.forward_backflow
            self.jastrow = self.omni.forward_jastrow

    @classmethod
    def from_pyscf(
        cls,
        mf,
        init_weights=True,
        freeze_mos=False,
        freeze_confs=False,
        conf_cutoff=1e-3,
        **kwargs,
    ):
        n_up, n_down = mf.mol.nelec
        try:
            conf_coeff, *confs = zip(
                *mf.fcisolver.large_ci(
                    mf.ci, mf.ncas, mf.nelecas, tol=conf_cutoff, return_strs=False
                )
            )
            conf_coeff = torch.tensor(conf_coeff)
            confs = (
                torch.tensor(confs, dtype=torch.long)
                .permute(1, 0, 2)
                .flatten(start_dim=1)
            )
        except AttributeError:
            confs = None
        geom = Geometry(mf.mol.atom_coords().astype('float32'), mf.mol.atom_charges())
        basis = GTOBasis.from_pyscf(mf.mol)
        wf = cls(geom, n_up, n_down, basis, configurations=confs, **kwargs)
        if init_weights:
            wf.mo.init_from_pyscf(mf, freeze_mos=freeze_mos)
            if confs is not None:
                wf.conf_coeff.weight.detach().copy_(conf_coeff)
                if freeze_confs:
                    wf.conf_coeff.weight.requires_grad_(False)
        return wf

    def forward(self, rs, debug=NULL_DEBUG):
        batch_dim, n_elec = rs.shape[:2]
        assert n_elec == self.confs.shape[1]
        n_atoms = len(self.coords)
        diffs_nuc = pairwise_diffs(
            torch.cat([self.coords, rs.flatten(end_dim=1)]), self.coords
        )
        edges_nuc = (
            self.dist_basis(diffs_nuc[:, :, 3].sqrt())
            if self.jastrow or self.mo.net
            else None
        )
        with debug.cd('mos'):
            xs = self.mo(diffs_nuc, edges_nuc, debug=debug)
        xs = debug['slaters'] = xs.view(batch_dim, n_elec, -1)
        if self.backflow or self.cusp_same or self.jastrow:
            dists_elec = pairwise_distance(rs, rs)
        if self.backflow or self.jastrow:
            edges_nuc = edges_nuc[n_atoms:].view(batch_dim, n_elec, n_atoms, -1)
            edges = torch.cat([self.dist_basis(dists_elec), edges_nuc], dim=2)
        if self.backflow:
            with debug.cd('backflow'):
                xs = self.backflow(xs, edges, debug=debug)
        conf_up, conf_down = self.confs[:, : self.n_up], self.confs[:, self.n_up :]
        det_up = debug['det_up'] = eval_slater(
            xs[:, : self.n_up, conf_up].permute(0, 2, 1, 3).flatten(end_dim=1)
        ).view(batch_dim, -1)
        det_down = debug['det_down'] = eval_slater(
            xs[:, self.n_up :, conf_down].permute(0, 2, 1, 3).flatten(end_dim=1)
        ).view(batch_dim, -1)
        psi = self.conf_coeff(det_up * det_down).squeeze(dim=-1)
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
            with debug.cd('jastrow'):
                psi = psi * torch.exp(self.jastrow(edges, debug=debug))
        return psi
