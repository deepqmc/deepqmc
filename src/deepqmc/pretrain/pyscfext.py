import logging
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional, Union

import jax
import jax.numpy as jnp
import pyscf.lib.chkfile as chk
from pyscf import gto
from pyscf.gto.basis import ALIAS as PYSCF_BASIS_SETS
from pyscf.mcscf import CASSCF
from pyscf.mcscf.df import _DFCAS, _DFCASCI, _DFCASSCF
from pyscf.scf import RHF

from ..hamil import MolecularHamiltonian
from ..molecule import Molecule
from .gto import GTOBasis

log = logging.getLogger(__name__)


def filter_string(string: str):
    """Removes dashes and underscores from string and returns all lower case."""

    return ''.join([s for s in string if s not in ['-', '_']]).lower()


def extend_basis(hamil: MolecularHamiltonian, basis: str) -> Mapping[int, str]:
    """Takes a basis set string and checks for compatibility with ECPs."""
    basis_dict = {}
    if any(hamil.ecp_mask):
        assert hamil.ecp_type is not None
        ecp_type = filter_string(hamil.ecp_type)
        basis = filter_string(basis)
        ecp_basis = ecp_type + basis
        if ecp_basis in PYSCF_BASIS_SETS.keys():
            basis_dict = {
                int(c): ecp_basis if m else basis
                for c, m in zip(hamil.mol.charges, hamil.ecp_mask)
            }
            log.info(f'{ecp_basis} employed for atoms with effective core potential')
        else:
            log.warning(
                f'No ecp variant of {basis} basis found, but ecps are used. This may'
                ' lead to an inaccurate pretraining target. Consider manually'
                ' specifying a basis set for each atom.'
            )
    if any(~hamil.ecp_mask) and ('ecp' in basis or 'bfd' in basis):
        log.warning(
            f'Using the {basis} for atoms without ECP may result in'
            ' inaccurate pretraining target.'
        )
    basis_dict = basis_dict or {int(c): basis for c in hamil.mol.charges}
    return basis_dict


def pyscf_from_hamil(  # type: ignore
    hamil: MolecularHamiltonian,
    basis: Union[str, Mapping[int, str]],
    coords: Optional[jax.Array] = None,
    n_states: int = 1,
    cas: Optional[tuple[int, int]] = None,
    state_avg: bool = True,
    fix_spin: Optional[float] = None,
    chkfile: Optional[str] = None,
    **kwargs,
):
    r"""Create a pyscf molecule and perform an SCF calculation on it.

    Args:
        hamil (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the
            molecule on which to perform the SCF calculation.
        basis (str): the name of the Gaussian basis set to use.
        coords (jax.Array): optional, nuclear coordinates differring from hamil.
        n_states (int): optional, the number of electronic states to compute.
        cas (tuple[int,int]): optional, the active space definition for CASSCF.
        state_avg (bool): optional, whether to use state averaging in CASSCF
            for excited states.
        fix_spin (int): optional, whether to target specific spin states
            (S^2 value) in CASSCF.
        workdir (str): optional, directory for storing pyscf results.

    Returns:
        tuple: the pyscf molecule and the SCF calculation object.
    """
    for atomic_number in hamil.mol.charges[jnp.invert(hamil.ecp_mask)].tolist():
        assert atomic_number not in hamil.mol.charges[hamil.ecp_mask], (
            'pyscf does not allow atoms of the same type to have different'
            ' configurations for their effective core potentials (i.e. w/wo).'
        )

    if isinstance(basis, str):
        basis = extend_basis(hamil, basis)

    mol = gto.M(
        **hamil.as_pyscf(coords=coords),
        basis=basis,
        cart=True,
        parse_arg=False,
        verbose=0,
        **kwargs,
    )
    log.info('Running HF...')
    mf = RHF(mol)
    mf.kernel()
    log.info(f'HF energy: {mf.e_tot}')
    mc = None
    if cas:
        log.info('Running MCSCF...')
        mc = CASSCF(mf, *cas)
        assert not isinstance(mc, _DFCAS | _DFCASCI | _DFCASSCF)
        if n_states > 1:
            mc.fcisolver.nroots = n_states
            if state_avg:
                mc.state_average_(jnp.ones(n_states) / n_states)
        if fix_spin is not None:
            mc.fcisolver.spin = fix_spin  # type: ignore
            mc.fix_spin_(ss=fix_spin)
        mc.kernel()
        log.info(f'MCSCF energy: {mc.fcisolver.eci}')
    if chkfile:
        assert mf.chkfile
        log.info(f'Dump PySCF checkpoint to {chkfile}')
        pyscf_chkfile = mf.chkfile  # mf and mc share chkfile
        if mc:
            chk.dump(pyscf_chkfile, 'ci', mc.ci)
            chk.dump(pyscf_chkfile, 'nelecas', mc.nelecas)
            chk.dump(pyscf_chkfile, 'fcisolver.eci', mc.fcisolver.eci)
        shutil.copy(pyscf_chkfile, chkfile)
    return mol, (mf, mc)


def pyscf_from_chkfile(chkfile: str, validate: Optional[dict] = None):
    r"""Recover PySCF solution from file.

    Args:
        chkfile (str): path to PySCF checkpoint.
        validate (dict): optional, kwargs to compare with the restored PySCF object.

    Returns:
        tuple: the pyscf molecule and the SCF calculation object.
    """
    assert Path(chkfile).is_file()
    log.info(f'Restoring PySCF object from {chkfile}')
    mol = chk.load_mol(chkfile)
    mf = RHF(mol)
    scf_data = chk.load(chkfile, 'scf')
    assert isinstance(scf_data, dict)
    mf.__dict__.update(scf_data)
    log.info(f'HF energy: {mf.e_tot}')
    mc_dict = chk.load(chkfile, 'mcscf')
    mc = None
    if mc_dict:
        mc = CASSCF(mf, 0, 0)
        assert not isinstance(mc, _DFCAS | _DFCASCI | _DFCASSCF)
        mc.__dict__.update(mc_dict)
        mc.ci = chk.load(chkfile, 'ci')
        nelecas = chk.load(chkfile, 'nelecas')
        assert isinstance(nelecas, tuple)
        mc.nelecas = (int(nelecas[0]), int(nelecas[1]))
        mc.fcisolver.eci = chk.load(chkfile, 'fcisolver.eci')
        log.info(f'MCSCF energy: {mc.fcisolver.eci}')
    if validate is not None:
        for key, val in validate.items():
            assert mol.__dict__[key] == val, (
                f'The specified {key} ({val}) does not match {key} found in checkpoint'
                f' ({mol.__dict__[key]})!'
            )
    return mol, (mf, mc)


def confs_from_mc(mc, tol=-1):
    r"""Retrieve the electronic configurations contributing to a pyscf CAS-SCF solution.

    Args:
        mc: a pyscf MC-SCF object.
        tol (float): default -1, the CI weight threshold, default value is negative
            to make sure that all determinants are included
            (even those with numerically zero weight).

            *mc.fcisolver.large_ci(
    Returns:
        list: the list of configurations in deepqmc format,
        with weight larger than :data:`tol`.
    """
    cis = mc.ci if isinstance(mc.ci, list) else [mc.ci]
    # This is required because pyscf returns a list only if multiple roots are specified
    state_conf_coeffs, state_confs = [], []
    for ci in cis:
        conf_coeff, *confs = zip(
            *mc.fcisolver.large_ci(ci, mc.ncas, mc.nelecas, tol=tol, return_strs=False)
        )
        conf_coeff = jnp.array(conf_coeff)
        sort_idxs = jnp.argsort(-(conf_coeff**2))
        confs = [
            [
                jnp.tile(jnp.arange(mc.ncore), (len(conf_coeff), 1)),
                jnp.array(cfs) + mc.ncore,
            ]
            for cfs in confs
        ]
        confs = jnp.concatenate(
            [jnp.concatenate(cfs, axis=-1) for cfs in confs], axis=-1
        )
        state_conf_coeffs.append(conf_coeff[sort_idxs])
        state_confs.append(confs[sort_idxs])
    return jnp.stack(state_conf_coeffs), jnp.stack(state_confs)


def compute_scf_solution(
    mols: Union[Molecule, list[Molecule]],
    hamil: MolecularHamiltonian,
    n_states: int,
    *,
    basis: str = '6-31G',
    cas: Optional[tuple[int, int]] = None,
    workdir: Optional[str] = None,
    **pyscf_kwargs,
):
    r"""Compute the SCF solutions for :data:`mols`.

    Args:
        mols (~deepqmc.Molecule): the molecule or a sequence of molecules to
            consider.
        hamil (~deepqmc.hamil.MolecularHamiltonian): the Hamiltonian of the
            system.
        n_states (int): the number of electronic states to consider.
        basis (str): the name of a Gaussian basis set.
        cas (tuple[int,int]): optional the active space specification for CAS-SCF.
    """

    mols = mols if isinstance(mols, Sequence) else [mols]

    chkpt_dir = f'{workdir}/pyscf_chkpts' if workdir else None
    restore = False
    if chkpt_dir:
        os.makedirs(chkpt_dir, exist_ok=True)
        restore = len(os.listdir(chkpt_dir)) == len(mols)
        assert restore or not os.listdir(chkpt_dir), (
            'The specified workdir contains pyscf checkpoints, which do not match'
            ' the number of molecules specified in mols.'
        )

    def get_pyscf(coords, chkfile):
        if restore:
            assert chkfile
            pyscf_hamil = hamil.as_pyscf(coords=coords)
            validate = {key: pyscf_hamil[key] for key in ['atom', 'charge', 'ecp']}
            validate['basis'] = extend_basis(hamil, basis)
            return pyscf_from_chkfile(chkfile, validate)
        else:
            return pyscf_from_hamil(
                hamil,
                basis,
                coords,
                n_states,
                cas=cas,
                chkfile=chkfile,
                **pyscf_kwargs,
            )

    mol_pyscf, mo_coeffs, confs, conf_coeffs = None, [], [], []
    for i, mol in enumerate(mols):
        chkfile = None if chkpt_dir is None else f'{chkpt_dir}/mol_{i}.pyscf_chkpt'
        mol_pyscf, (mf, mc) = get_pyscf(mol.coords, chkfile)
        mo_coeffs_i = jnp.asarray(mc.mo_coeff if mc else mf.mo_coeff)
        ao_overlap = jnp.asarray(mf.mol.intor('int1e_ovlp_cart'))
        mo_coeffs_i *= jnp.sqrt(jnp.diag(ao_overlap))[:, None]
        conf_coeffs_i, confs_i = (
            confs_from_mc(mc)
            if mc
            else (
                jnp.ones((n_states, 1)),
                jnp.array(
                    [[list(range(hamil.n_up)) + list(range(hamil.n_down))]] * n_states
                ),
            )
        )
        mo_coeffs.append(mo_coeffs_i)
        confs.append(confs_i)
        conf_coeffs.append(conf_coeffs_i)

    centers, shells = GTOBasis.from_pyscf(mol_pyscf)  # all molecules share the basis

    return {
        'centers': centers,
        'shells': shells,
        'mo_coeffs': jnp.stack(mo_coeffs),
        'confs': jnp.stack(confs).swapaxes(0, 1),
        'conf_coeffs': jnp.stack(conf_coeffs).swapaxes(0, 1),
    }
