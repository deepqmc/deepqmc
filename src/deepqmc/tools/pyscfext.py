import numpy as np
import pyscf.dft.numint

__all__ = ()


def eval_ao_normed(mol, *args, **kwargs):
    aos = pyscf.dft.numint.eval_ao(mol, *args, **kwargs)
    if mol.cart:
        aos /= np.sqrt(np.diag(mol.intor('int1e_ovlp_cart')))
    return aos


def electron_density_of(mf, rs):
    aos = eval_ao_normed(mf.mol, rs)
    return pyscf.dft.numint.eval_rho2(mf.mol, aos, mf.mo_coeff, mf.mo_occ, xctype='LDA')
