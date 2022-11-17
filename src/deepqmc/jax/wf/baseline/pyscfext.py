import logging

import numpy as jnp
from pyscf import gto
from pyscf.mcscf import CASSCF
from pyscf.scf import RHF

log = logging.getLogger(__name__)


def pyscf_from_mol(mol, basis, cas=None):
    mol = gto.M(
        atom=mol.as_pyscf(),
        unit='bohr',
        basis=basis,
        charge=mol.charge,
        spin=mol.spin,
        cart=True,
        parse_arg=False,
    )
    log.info('Running HF...')
    mf = RHF(mol)
    mf.kernel()
    if cas:
        log.info('Running MCSCF...')
        mc = CASSCF(mf, *cas)
        mc.kernel()
    return mf, mc if cas else None


def confs_from_mc(mc, tol=0):
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
