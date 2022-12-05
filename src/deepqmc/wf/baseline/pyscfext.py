import logging

import numpy as jnp
from pyscf import gto
from pyscf.mcscf import CASSCF
from pyscf.scf import RHF

log = logging.getLogger(__name__)


def pyscf_from_mol(mol, basis, cas=None, **kwargs):
    r"""Create a pyscf molecule and perform an SCF calculation on it.

    Args:
        mol (~deepqmc.Molecule): the molecule on which to perform the SCF calculation.
        basis (str): the name of the Gaussian basis set to use.
        cas (Tuple[int,int]): optional, the active space definition for CAS.

    Returns:
        tuple: the pyscf molecule and the SCF calculation object.
    """
    mol = gto.M(
        atom=mol.as_pyscf(),
        unit='bohr',
        basis=basis,
        charge=mol.charge,
        spin=mol.spin,
        cart=True,
        parse_arg=False,
        **kwargs,
    )
    log.info('Running HF...')
    mf = RHF(mol)
    mf.kernel()
    if cas:
        log.info('Running MCSCF...')
        mc = CASSCF(mf, *cas)
        mc.kernel()
    return mol, (mf, mc if cas else None)


def confs_from_mc(mc, tol=0):
    r"""Retrieve the electronic configurations contributing to a pyscf CAS-SCF solution.

    Args:
        mc: a pyscf MC-SCF object.
        tol (float): default 0, the CI weight threshold.

    Returns:
        list: the list of configurations in deepqmc format,
        with weight larger than :data:`tol`.
    """
    conf_coeff, *confs = zip(
        *mc.fcisolver.large_ci(mc.ci, mc.ncas, mc.nelecas, tol=tol, return_strs=False)
    )
    confs = [
        [
            jnp.tile(jnp.arange(mc.ncore), (len(conf_coeff), 1)),
            jnp.array(cfs) + mc.ncore,
        ]
        for cfs in confs
    ]
    confs = jnp.concatenate([jnp.concatenate(cfs, axis=-1) for cfs in confs], axis=-1)
    confs = sorted(zip(conf_coeff, confs), key=lambda x: -x[0] ** 2)
    return confs
