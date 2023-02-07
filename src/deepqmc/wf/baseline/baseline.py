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

    def __init__(self, mol, centers, shells, mo_coeffs, confs, conf_coeffs):
        super().__init__(mol)
        self.basis = GTOBasis(centers, shells)
        mo_coeffs = jnp.asarray(mo_coeffs)
        self.mo_coeffs = hk.get_parameter(
            'mo_coeffs', mo_coeffs.shape, init=lambda s, d: mo_coeffs
        )
        self.conf_coeffs = hk.get_parameter(
            'conf_coeffs', conf_coeffs.shape, init=lambda s, d: conf_coeffs
        )
        self.confs = confs

    def __call__(self, phys_conf, return_mos=False):
        diffs = pairwise_diffs(phys_conf.r, phys_conf.R)
        aos = self.basis(diffs)
        mos = jnp.einsum(
            '...mo,...em->...eo', self.mo_coeffs[phys_conf.config_idx], aos
        )
        confs = self.confs[phys_conf.config_idx]
        conf_up, conf_down = confs[..., : self.n_up], confs[..., self.n_up :]
        det_up = mos[..., : self.n_up, conf_up].swapaxes(-2, -3)
        det_down = mos[..., self.n_up :, conf_down].swapaxes(-2, -3)
        if return_mos:
            return det_up, det_down
        sign_up, det_up = eval_log_slater(det_up)
        sign_down, det_down = eval_log_slater(det_down)
        dets = sign_up * sign_down * jnp.exp(det_up + det_down)
        psi = jnp.einsum(
            '...id,...d->...i', self.conf_coeffs[phys_conf.config_idx], dets
        ).squeeze(axis=-1)
        return Psi(jnp.sign(psi), jnp.log(jnp.abs(psi)))

    @classmethod
    def from_mol(cls, mol, Rs, *, basis='6-31G', cas=None, **kwargs):
        r"""Create input to the constructor from a :class:`~deepqmc.Molecule`.

        Args:
            mol (~deepqmc.Molecule): the molecule to consider.
            basis (str): the name of a Gaussian basis set.
            cas (Tuple[int,int]): optional the active space specification for CAS-SCF.
        """
        Rs = [Rs] if isinstance(Rs, jnp.ndarray) and Rs.ndim == 2 else Rs
        mo_coeffs, confs, conf_coeffs = [], [], []
        for R in Rs:
            mol_pyscf, (mf, mc) = pyscf_from_mol(mol, R, basis, cas, **kwargs)
            centers, shells = GTOBasis.from_pyscf(mol_pyscf)
            mo_coeff = jnp.asarray(mc.mo_coeff if mc else mf.mo_coeff)
            ao_overlap = jnp.asarray(mf.mol.intor('int1e_ovlp_cart'))
            mo_coeff *= jnp.sqrt(jnp.diag(ao_overlap))[:, None]
            conf_coeff, conf = (
                ([1], [sum([list(range(n_el)) for n_el in (mol.n_up, mol.n_down)], [])])
                if mc is None
                else zip(*confs_from_mc(mc))
            )
            conf_coeff, conf = jnp.array(conf_coeff).reshape(-1, 1), jnp.array(conf)
            mo_coeffs.append(mo_coeff)
            confs.append(conf)
            conf_coeffs.append(conf_coeff)
        return (
            centers,
            shells,
            jnp.stack(mo_coeffs),
            jnp.stack(confs),
            jnp.stack(conf_coeffs),
        )
