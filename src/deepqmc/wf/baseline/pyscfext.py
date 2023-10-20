import logging

import jax.numpy as jnp
from pyscf import gto
from pyscf.mcscf import CASSCF
from pyscf.scf import RHF

log = logging.getLogger(__name__)


def pyscf_from_mol(hamil, coords, basis, cas=None, **kwargs):
    r"""Create a pyscf molecule and perform an SCF calculation on it.

    Args:
        hamil (~deepqmc.hamil.qc.MolecularHamiltonian): the Hamiltonian of the
            molecule on which to perform the SCF calculation.
        coords (jax.Array): the nuclear coordinates of the molecule, shape: [n_nuc, 3].
        basis (str): the name of the Gaussian basis set to use.
        cas (Tuple[int,int]): optional, the active space definition for CAS.

    Returns:
        tuple: the pyscf molecule and the SCF calculation object.
    """
    for atomic_number in hamil.mol.charges[jnp.invert(hamil.pp_mask)].tolist():
        assert atomic_number not in hamil.mol.charges[hamil.pp_mask], (
            'Usage of different pseudopotentials for atoms of the same element is not'
            ' implemented for pretraining.'
        )
    mol = gto.M(
        atom=hamil.as_pyscf(coords),
        unit='bohr',
        basis=basis,
        charge=hamil.mol.charge,
        spin=hamil.mol.spin,
        cart=True,
        parse_arg=False,
        ecp={int(charge): hamil.pp_type for charge in hamil.mol.charges[hamil.pp_mask]},
        verbose=0,
        **kwargs,
    )
    log.info('Running HF...')
    mf = RHF(mol)
    mf.kernel()
    log.info(f'HF energy: {mf.e_tot}')
    if cas:
        log.info('Running MCSCF...')
        mc = CASSCF(mf, *cas)
        mc.kernel()
        log.info(f'MCSCF energy: {mc.e_tot}')
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
