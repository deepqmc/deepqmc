from functools import partial

import numpy as np
import torch
from torch import nn

from ..geom import Geometry
from ..utils import NULL_DEBUG
from .anti import eval_slater
from .base import BaseWFNet
from .gto import GTOBasis


class HFNet(BaseWFNet):
    def __init__(self, geom, n_up, n_down, basis, mo_factory=None):
        mo_factory = mo_factory or partial(nn.Linear, bias=False)
        super().__init__()
        self.n_up, self.n_down = n_up, n_down
        self.register_geom(geom)
        self.basis = basis
        self.mo = mo_factory(basis.dim, max(n_up, n_down))

    def init_from_pyscf(self, mf):
        assert isinstance(self.mo, nn.Linear)
        mo_coeff = mf.mo_coeff.copy()
        if mf.mol.cart:
            mo_coeff *= np.sqrt(np.diag(mf.mol.intor('int1e_ovlp_cart')))[:, None]
        self.mo.weight.detach().copy_(
            torch.from_numpy(mo_coeff[:, : max(self.n_up, self.n_down)].T)
        )

    @classmethod
    def from_pyscf(cls, mf):
        n_up = (mf.mo_occ >= 1).sum()
        n_down = (mf.mo_occ == 2).sum()
        assert (mf.mo_occ[:n_down] == 2).all()
        assert (mf.mo_occ[n_down:n_up] == 1).all()
        assert (mf.mo_occ[n_up:] == 0).all()
        geom = Geometry(mf.mol.atom_coords().astype('float32'), mf.mol.atom_charges())
        basis = GTOBasis.from_pyscf(mf.mol)
        wf = cls(geom, n_up, n_down, basis)
        wf.init_from_pyscf(mf)
        return wf

    def forward(self, rs, debug=NULL_DEBUG):
        batch_dim, n_elec = rs.shape[:2]
        xs = debug['aos'] = self.basis(rs.flatten(end_dim=1)).view(
            batch_dim, n_elec, -1
        )
        xs = debug['slaters'] = self.mo(xs)
        det_up = debug['det_up'] = eval_slater(xs[:, : self.n_up, : self.n_up])
        det_down = debug['det_down'] = eval_slater(xs[:, self.n_up :, : self.n_down])
        return det_up * det_down

    def orbitals(self, rs):
        return self.mo(self.basis(rs))

    def density(self, rs):
        xs = self.orbitals(rs)
        return sum(
            (xs[:, :n_elec] ** 2).sum(dim=-1) for n_elec in (self.n_up, self.n_down)
        )
