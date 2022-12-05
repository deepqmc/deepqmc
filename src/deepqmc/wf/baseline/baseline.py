import haiku as hk
import jax.numpy as jnp

from ...physics import pairwise_diffs
from ...types import Psi
from ..base import WaveFunction
from .gto import GTOBasis
from .pyscfext import confs_from_mc, pyscf_from_mol

__all__ = ['Baseline']


def eval_log_slater(xs):
    if xs.shape[-1] == 0:
        return jnp.ones(xs.shape[:-2]), jnp.zeros(xs.shape[:-2])
    return jnp.linalg.slogdet(xs)


class Baseline(WaveFunction):
    r"""Represent an (MC-)SCF wave function, used as baseline."""

    def __init__(self, mol, centers, shells, mo_coeff, confs, conf_coeff):
        super().__init__(mol)
        self.basis = GTOBasis(centers, shells)
        mo_coeff = jnp.asarray(mo_coeff)
        self.mo_coeff = hk.Linear(
            mo_coeff.shape[-1],
            with_bias=False,
            w_init=lambda s, d: mo_coeff,
            name='mo_coeff',
        )
        self.conf_coeff = hk.Linear(
            1, with_bias=False, w_init=lambda s, d: conf_coeff, name='conf_coeff'
        )
        self.confs = confs

    def __call__(self, r, return_mos=False):
        diffs = pairwise_diffs(r, self.mol.coords)
        aos = self.basis(diffs)
        mos = self.mo_coeff(aos)
        conf_up, conf_down = self.confs[..., : self.n_up], self.confs[..., self.n_up :]
        det_up = mos[..., : self.n_up, conf_up].swapaxes(-2, -3)
        det_down = mos[..., self.n_up :, conf_down].swapaxes(-2, -3)
        if return_mos:
            return det_up, det_down
        sign_up, det_up = eval_log_slater(det_up)
        sign_down, det_down = eval_log_slater(det_down)
        psi = self.conf_coeff(
            sign_up * sign_down * jnp.exp(det_up + det_down)
        ).squeeze()
        return Psi(jnp.sign(psi), jnp.log(jnp.abs(psi)))

    @classmethod
    def from_mol(cls, mol, *, basis='6-31G', cas=None, **kwargs):
        r"""Create input to the constructor from a :class:`~deepqmc.Molecule`.

        Args:
            mol (~deepqmc.Molecule): the molecule to consider.
            basis (str): the name of a Gaussian basis set.
            cas (Tuple[int,int]): optional the active space specification for CAS-SCF.
        """
        mol_pyscf, (mf, mc) = pyscf_from_mol(mol, basis, cas, **kwargs)
        centers, shells = GTOBasis.from_pyscf(mol_pyscf)
        mo_coeff = jnp.asarray(mc.mo_coeff if mc else mf.mo_coeff)
        ao_overlap = jnp.asarray(mf.mol.intor('int1e_ovlp_cart'))
        mo_coeff *= jnp.sqrt(jnp.diag(ao_overlap))[:, None]
        conf_coeff, confs = (
            ([1], [sum([list(range(n_el)) for n_el in (mol.n_up, mol.n_down)], [])])
            if mc is None
            else zip(*confs_from_mc(mc))
        )
        conf_coeff, confs = jnp.array(conf_coeff).reshape(-1, 1), jnp.array(confs)
        return centers, shells, mo_coeff, confs, conf_coeff
