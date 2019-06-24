import torch

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


class HFNet(BaseWFNet):
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
        cusp_correction=True,
        cusp_electrons=True,
        rc_scaling=1.0,
    ):
        super().__init__()
        self.n_up, self.n_down = n_up, n_down
        self.register_geom(geom)
        self.dist_basis = (
            DistanceBasis(dist_basis_dim, cutoff=dist_basis_cutoff, envelope='nocusp')
            if mo_factory or jastrow_factory
            else None
        )
        self.mo = MolecularOrbital(
            geom,
            basis,
            max(n_up, n_down),
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

    def init_from_pyscf(self, mf, **kwargs):
        self.mo.init_from_pyscf(mf, **kwargs)

    @classmethod
    def from_pyscf(cls, mf, init_mos=True, freeze_mos=False, **kwargs):
        n_up = (mf.mo_occ >= 1).sum()
        n_down = (mf.mo_occ == 2).sum()
        assert (mf.mo_occ[:n_down] == 2).all()
        assert (mf.mo_occ[n_down:n_up] == 1).all()
        assert (mf.mo_occ[n_up:] == 0).all()
        geom = Geometry(mf.mol.atom_coords().astype('float32'), mf.mol.atom_charges())
        basis = GTOBasis.from_pyscf(mf.mol)
        wf = cls(geom, n_up, n_down, basis, **kwargs)
        if init_mos:
            wf.init_from_pyscf(mf, freeze_mos=freeze_mos)
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
        xs = debug['mos'] = self.mo(diffs_nuc, edges_nuc, debug=debug)
        xs = debug['slaters'] = xs.view(batch_dim, n_elec, -1)
        det_up = debug['det_up'] = eval_slater(xs[:, : self.n_up, : self.n_up])
        det_down = debug['det_down'] = eval_slater(xs[:, self.n_up :, : self.n_down])
        psi = det_up * det_down
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
