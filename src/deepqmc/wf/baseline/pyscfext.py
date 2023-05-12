import logging

import jax.numpy as jnp
from pyscf import gto
from pyscf.mcscf import CASSCF
from pyscf.scf import RHF

from ...utils import pad_list_of_3D_arrays_to_one_array

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
    for atomic_number in mol.charges[jnp.invert(mol.pp_mask)].tolist():
        assert atomic_number not in mol.charges[mol.pp_mask], (
            'Usage of different pseudopotentials for atoms of the same element is not'
            ' implemented for pretraining.'
        )
    mol = gto.M(
        atom=mol.as_pyscf(),
        unit='bohr',
        basis=basis,
        charge=mol.charge,
        spin=mol.spin,
        cart=True,
        parse_arg=False,
        ecp={int(charge): mol.pp_type for charge in mol.charges[mol.pp_mask]},
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


def parse_pp_params(mol):
    """Load ane parse the pseudopotential parameters from the pyscf package.

    This function loads the pseudopotential parameters for an atom (given by `charge`
    argument) from the pyscf package and parses them to jnp arrays.

    Args:
        mol (~deepqmc.molecule.Molecule): the molecule to consider
    Returns:
        tuple: a tuple containing a an array of integers indicating the numbers of core
            electrons replaced by pseudopotential, an array of local pseudopotential
            parameters (padded by zeros if each atom has a different shape of local
            parameters), and an array of nonlocal pseudopotential parameters (also
            padded by zeros).
    """

    ns_core, pp_loc_params, pp_nl_params = [], [], []
    max_number_of_same_type_terms = []
    for i, atomic_number in enumerate(mol.charges):
        if mol.pp_mask[i]:
            _, data = gto.M(
                atom=[(int(atomic_number), jnp.array([0, 0, 0]))],
                spin=atomic_number % 2,
                basis='6-31G',
                ecp=mol.pp_type,
            )._ecp.popitem()
            pp_loc_param = data[1][0][1][1:4]
            if data[0] != 0:
                pp_nl_param = jnp.array([di[1][2] for di in data[1][1:]]).swapaxes(
                    -1, -2
                )
            else:
                pp_nl_param = jnp.array([[[]]])

            max_number_of_same_type_terms.append(len(max(pp_loc_param, key=len)))
            n_core = data[0]
        else:
            n_core = 0
            pp_loc_param = [[], [], []]
            pp_nl_param = jnp.asarray([[[]]])
        ns_core.append(n_core)
        pp_loc_params.append(pp_loc_param)
        pp_nl_params.append(pp_nl_param)

    ns_core = jnp.asarray(ns_core)

    # We need to pad local parameters with zeros to be able to
    # convert pp_loc params from list to jnp.array.
    pad = max(max_number_of_same_type_terms, default=0)
    pp_loc_param_padded = []
    for pp_loc_param in pp_loc_params:
        pp_loc_param = [pi + [[0, 0]] * (pad - len(pi)) for pi in pp_loc_param]
        pp_loc_param_padded.append(jnp.swapaxes(jnp.array(pp_loc_param), -1, -2))
        # shape (r^n term, coefficient (β) & exponent (α), no. of terms with the same n)
    pp_loc_params = jnp.array(pp_loc_param_padded)

    # We also pad the non-local parameters with zeros
    pp_nl_params = pad_list_of_3D_arrays_to_one_array(pp_nl_params)

    return ns_core, pp_loc_params, jnp.array(pp_nl_params)
