import haiku as hk
import jax.numpy as jnp
from pyscf import gto, scf

from ...types import Psi
from ...utils import pairwise_diffs
from ..base import WaveFunction
from .gto import GTOBasis

__all__ = ['HartreeFock']


def eval_log_slater(xs):
    if xs.shape[-1] == 0:
        return jnp.ones(xs.shape[:-2]), jnp.zeros(xs.shape[:-2])
    return jnp.linalg.slogdet(xs)


class HartreeFock(WaveFunction):
    def __init__(self, mol, centers, shells, mo_coeff):
        super().__init__(mol)
        self.basis = GTOBasis(centers, shells)
        mo_coeff = jnp.asarray(mo_coeff)
        self.mo_coeff = hk.Linear(
            mo_coeff.shape[-1], with_bias=False, w_init=lambda s, d: mo_coeff
        )
        self.confs = jnp.array([list(range(self.n_up)) + list(range(self.n_down))])

    def __call__(self, r):
        diffs = pairwise_diffs(r, self.mol.coords)
        aos = self.basis(diffs)
        mos = self.mo_coeff(aos)
        conf_up, conf_down = self.confs[..., : self.n_up], self.confs[..., self.n_up :]
        det_up = mos[..., : self.n_up, conf_up].squeeze(axis=-2)
        det_down = mos[..., self.n_up :, conf_down].squeeze(axis=-2)
        sign_up, det_up = eval_log_slater(det_up)
        sign_down, det_down = eval_log_slater(det_down)
        sign_psi, log_psi = sign_up * sign_down, det_up + det_down
        return Psi(sign_psi, log_psi)

    @classmethod
    def from_mol(cls, mol, basis='6-31G'):
        mol_pyscf = gto.M(
            atom=mol.as_pyscf(),
            unit='bohr',
            basis=basis,
            charge=mol.charge,
            spin=mol.spin,
            cart=True,
        )
        centers, shells = GTOBasis.from_pyscf(mol_pyscf)
        mf = scf.RHF(mol_pyscf)
        mf.kernel()
        mo_coeff = jnp.asarray(mf.mo_coeff)
        ao_overlap = jnp.asarray(mf.mol.intor('int1e_ovlp_cart'))
        mo_coeff *= jnp.sqrt(jnp.diag(ao_overlap))[:, None]
        return centers, shells, mo_coeff
