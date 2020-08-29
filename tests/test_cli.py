import os

import pytest
import toml
from click.testing import CliRunner

from deepqmc.cli import cli
from deepqmc.errors import TomlError

PARAM_H2 = {
    'system': 'H2',
    'ansatz': 'paulinet',
    'train_kwargs': {'n_steps': 0, 'equilibrate': False},
    'evaluate_kwargs': {'n_steps': 0, 'sample_kwargs': {'equilibrate': False}},
}

runner = CliRunner()


def test_defaults():
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['defaults'], catch_exceptions=False)
    assert toml.loads(result.output)


def test_train():
    with runner.isolated_filesystem():
        with open('param.toml', 'w') as f:
            toml.dump(PARAM_H2, f)
        result = runner.invoke(cli, ['train', '.', '--no-cuda'], catch_exceptions=False)
        files = os.listdir()
    assert 'fit.h5' in files
    assert 'pyscf.chk' in files
    assert 'chkpts' in files
    assert any(f.startswith('events.out.tfevents.') for f in files)
    assert 'converged SCF energy' in result.output


def test_pyscf_reload():
    with runner.isolated_filesystem():
        with open('param.toml', 'w') as f:
            toml.dump({**PARAM_H2, 'paulinet_kwargs': {'cas': [2, 2]}}, f)
        result = runner.invoke(cli, ['train', '.', '--no-cuda'], catch_exceptions=False)
        result_repeated = runner.invoke(
            cli, ['train', '.', '--no-cuda'], catch_exceptions=False
        )
    assert 'converged SCF energy' in result.output
    assert not result_repeated.output


def test_evaluate():
    with runner.isolated_filesystem():
        with open('param.toml', 'w') as f:
            toml.dump(PARAM_H2, f)
        result = runner.invoke(
            cli, ['evaluate', '.', '--no-cuda'], catch_exceptions=False
        )
        files = os.listdir()
    assert 'sample.h5' in files
    assert 'pyscf.chk' in files
    assert any(f.startswith('events.out.tfevents.') for f in files)
    assert 'converged SCF energy' in result.output


def test_validity_check():
    with runner.isolated_filesystem():
        with open('param.toml', 'w'):
            pass
        with pytest.raises(TomlError):
            runner.invoke(cli, ['train', '.', '--no-cuda'], catch_exceptions=False)
        with open('param.toml', 'w') as f:
            toml.dump({**PARAM_H2, 'foo': 'bar'}, f)
        with pytest.raises(TomlError):
            runner.invoke(cli, ['train', '.', '--no-cuda'], catch_exceptions=False)
