from functools import partial
from typing import Sequence

import haiku as hk
import jax.numpy as jnp
from jax.nn import one_hot

from ...physics import pairwise_diffs
from ..base import WaveFunction
from .gto import GTOBasis
from .pyscfext import confs_from_mc, pyscf_from_mol

__all__ = ['Baseline']


class Baseline(WaveFunction):
    r"""Represent an (MC-)SCF wave function, used as baseline."""

    def __init__(
        self,
        hamil,
        n_determinants,
        centers,
        shells,
        mo_coeffs,
        confs,
        conf_coeffs,
        trainable=False,
    ):
        super().__init__(hamil)
        self.basis = GTOBasis(centers, shells)
        conf_coeffs = conf_coeffs[:, :n_determinants]
        self.mo_coeffs = (
            hk.get_parameter('mo_coeffs', mo_coeffs.shape, init=lambda s, d: mo_coeffs)
            if trainable
            else mo_coeffs
        )
        self.conf_coeffs = (
            hk.get_parameter(
                'conf_coeffs', conf_coeffs.shape, init=lambda s, d: conf_coeffs
            )
            if trainable
            else conf_coeffs
        )
        self.confs = confs[:, :n_determinants]

    def __call__(self, phys_conf):
        mol_idx = phys_conf.mol_idx
        diffs = pairwise_diffs(phys_conf.r, phys_conf.R)
        n_el = diffs.shape[-3]
        aos = self.basis(diffs)
        mos = jnp.einsum('...mo,...em->...eo', self.mo_coeffs[mol_idx], aos)
        mos = mos[:, self.confs[mol_idx]].swapaxes(-2, -3)
        # ci coefficients are included in the orbitals of the respective determinant
        factors = (jnp.abs(self.conf_coeffs[mol_idx]) ** (1 / n_el))[:, None] * (
            one_hot(0, n_el)[None, :] * jnp.sign(self.conf_coeffs[mol_idx])[:, None]
            + (1 - one_hot(0, n_el)[None, :])
        )
        return mos * factors[:, None, :]

    @classmethod
    def from_mol(
        cls, mols, hamil, *, basis='6-31G', cas=None, is_baseline=True, **pyscf_kwargs
    ):
        r"""Create input to the constructor from a :class:`~deepqmc.Molecule`.

        Args:
            mol (~deepqmc.Molecule): the molecule or a sequence of molecules to
                consider.
            basis (str): the name of a Gaussian basis set.
            cas (Tuple[int,int]): optional the active space specification for CAS-SCF.
            is_baseline (bool): dummy argument to indicate to the CLI that this class
                requires instantiation, due to interplay of haiku and pyscf. See
                :class:`~deepqmc.app.instantiate_ansatz` for the custom instantiation.
        """
        mols = mols if isinstance(mols, Sequence) else [mols]
        mo_coeffs, confs, conf_coeffs = [], [], []
        for mol in mols:
            mol_pyscf, (mf, mc) = pyscf_from_mol(
                hamil, mol.coords, basis, cas, **pyscf_kwargs
            )
            centers, shells = GTOBasis.from_pyscf(mol_pyscf)
            mo_coeff = jnp.asarray(mc.mo_coeff if mc else mf.mo_coeff)
            ao_overlap = jnp.asarray(mf.mol.intor('int1e_ovlp_cart'))
            mo_coeff *= jnp.sqrt(jnp.diag(ao_overlap))[:, None]
            conf_coeff, conf = (
                (
                    [1],
                    [
                        sum(
                            [list(range(n_el)) for n_el in (hamil.n_up, hamil.n_down)],
                            [],
                        )
                    ],
                )
                if mc is None
                else zip(*confs_from_mc(mc))
            )
            mo_coeffs.append(mo_coeff)
            confs.append(jnp.array(conf))
            conf_coeffs.append(jnp.array(conf_coeff))
        return partial(
            cls,
            **{
                'centers': centers,
                'shells': shells,
                'mo_coeffs': jnp.stack(mo_coeffs),
                'confs': jnp.stack(confs),
                'conf_coeffs': jnp.stack(conf_coeffs),
            },
        )
