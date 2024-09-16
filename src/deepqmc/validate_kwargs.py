import logging

import jax
from hydra.utils import call

log = logging.getLogger(__name__)


def validate_pretrain_kwargs(cfg):
    """Catch the most common misconfigurations in pretraining."""

    if (
        cfg.get('fit_fn', {})
        .get('loss_function_factory', {})
        .get('spin_penalty', False)
        and cfg.get('pretrain_steps', False)
        and cfg['pretrain_kwargs']['scf_kwargs'].get('cas', False)
        and not cfg['pretrain_kwargs']['scf_kwargs'].get('fix_spin', False)
    ):
        log.warning(
            'Variational training involves spin penalty. Consider adding the fix_spin'
            ' argument for the calculation of the pyscf baseline employed for the'
            ' pretraining.'
        )

    if cfg.get('electronic_states', 1) > 1 and (
        not cfg.get('pretrain_kwargs', False)
        or not cfg['pretrain_kwargs']['scf_kwargs'].get('cas', None)
    ):
        log.warning(
            'No CAS specified, all electronic states '
            'will be pretrained to HF ground state.'
        )


def validate_batch_size(cfg):
    """Make sure that batch sizes are acceptable in coputational setup."""

    assert not cfg.get('electron_batch_size', 0) % jax.device_count(), (
        f'Electron batch size ({cfg.get("electron_batch_size")}) cannot be '
        f'evenly split across {jax.device_count()} devices!'
    )
    mols = (
        call(cfg.get('mols')) if isinstance(cfg.get('mols'), dict) else cfg.get('mols')
    )
    len_mols = len(mols) if mols is not None else 1
    assert cfg.get('molecule_batch_size', 0) <= len_mols, (
        f'Molecule batch size ({cfg.get("molecule_batch_size")}) is larger than '
        f'the number of molecules in the dataset ({len_mols})!'
    )


def validate_kwargs(cfg):
    """Check that the combinations of configuration options are sensible."""

    validate_pretrain_kwargs(cfg)
    validate_batch_size(cfg)
