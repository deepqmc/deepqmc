import logging
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from hydra.utils import call
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def validate_pretrain_kwargs(cfg: DictConfig) -> None:
    """Catch the most common misconfigurations in pretraining."""

    loss_kwargs = cfg.get('loss_function_factory') or {}
    scf_kwargs = (cfg.get('pretrain_kwargs') or {}).get('scf_kwargs') or {}
    if (
        loss_kwargs.get('spin_penalty')
        and cfg.get('pretrain_steps')
        and scf_kwargs.get('cas')
        and scf_kwargs.get('fix_spin') is None
    ):
        log.warning(
            'Variational training involves spin penalty. Consider adding the fix_spin'
            ' argument for the calculation of the pyscf baseline employed for the'
            ' pretraining.'
        )

    if (
        cfg.get('electronic_states', 1) > 1
        and cfg.get('pretrain_steps', 0) > 0
        and (
            not cfg.get('pretrain_kwargs', False)
            or not cfg['pretrain_kwargs']['scf_kwargs'].get('cas', None)
        )
    ):
        log.warning(
            'No CAS specified, all electronic states '
            'will be pretrained to HF ground state.'
        )


def validate_overlap_scaling(cfg: DictConfig) -> None:
    """Warn if the energy-based overlap scaling or state sorting has no energies."""

    loss_kwargs = cfg.get('loss_function_factory') or {}
    energy_options = [
        option
        for option in ('scale_overlap_by', 'sort_states_by')
        if loss_kwargs.get(option)
    ]
    monitors = cfg.get('observable_monitors') or []
    has_energy_monitor = any(
        monitor == 'local_energy'
        or (
            isinstance(monitor, Mapping)
            and str(monitor.get('_target_', '')).endswith('EnergyMonitor')
        )
        for monitor in monitors
    )
    if energy_options and not has_energy_monitor:
        log.warning(
            f'{" and ".join(energy_options)} relies on the energy moving averages, '
            'which are only updated with a local_energy observable monitor. Add '
            'local_energy to observable_monitors, otherwise the default values '
            'are used.'
        )


def validate_batch_size(cfg: DictConfig) -> None:
    """Make sure that batch sizes are acceptable in coputational setup."""

    assert not cfg.get('electron_batch_size', 0) % jax.device_count(), (
        f'Electron batch size ({cfg.get("electron_batch_size")}) cannot be '
        f'evenly split across {jax.device_count()} devices!'
    )
    mols = (
        call(cfg.get('mols'), verbose=False)
        if isinstance(cfg.get('mols'), dict)
        else cfg.get('mols')
    )
    len_mols = len(mols) if mols is not None else 1
    assert cfg.get('molecule_batch_size', 0) <= len_mols or len_mols == 1, (
        f'Molecule batch size ({cfg.get("molecule_batch_size")}) is larger than '
        f'the number of molecules in the dataset ({len_mols})!'
    )


def validate_mols_consistency(cfg: DictConfig) -> None:
    """Make sure that ``mols`` is compatible with the Hamiltonian's molecule.

    The molecules in ``mols`` and the molecule specified in ``hamil.mol`` are
    assumed to be equivalent up to their geometry, i.e. they must share the same
    charges, total charge and spin.
    """

    mols = (
        call(cfg.get('mols'), verbose=False)
        if isinstance(cfg.get('mols'), dict)
        else cfg.get('mols')
    )
    if not mols:
        return
    hamil_mol = call(cfg['hamil']['mol'])
    for i, mol in enumerate(mols):
        assert jnp.array_equal(mol.charges, hamil_mol.charges), (
            f'Molecule {i} in `mols` has charges {mol.charges}, incompatible with '
            f'the charges {hamil_mol.charges} of `hamil.mol`!'
        )
        assert mol.charge == hamil_mol.charge, (
            f'Molecule {i} in `mols` has total charge {mol.charge}, incompatible '
            f'with the total charge {hamil_mol.charge} of `hamil.mol`!'
        )
        assert mol.spin == hamil_mol.spin, (
            f'Molecule {i} in `mols` has spin {mol.spin}, incompatible with the '
            f'spin {hamil_mol.spin} of `hamil.mol`!'
        )


def validate_spin_polarization(cfg: DictConfig) -> None:
    """Warn about fully spin-polarized systems, which are not supported."""

    hamil = call(cfg['hamil'])
    if hamil.n_down == 0:
        log.warning(
            f'The system has no spin-down electrons (n_up={hamil.n_up}, '
            'n_down=0). Fully spin-polarized systems are not supported and the '
            'calculation is expected to fail.'
        )


def validate_kwargs(cfg: DictConfig) -> None:
    """Check that the combinations of configuration options are sensible."""

    validate_spin_polarization(cfg)
    validate_pretrain_kwargs(cfg)
    validate_overlap_scaling(cfg)
    validate_batch_size(cfg)
    validate_mols_consistency(cfg)
