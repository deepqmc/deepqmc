import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skip(reason='TODO: resolve cublas error in pytest subprocss')
class TestApp:
    ARGS = [
        'deepqmc',
        'hamil/mol=H2',
        'device=cpu',
        'task.steps=1',
        'task.sample_size=2',
        '+task.max_eq_steps=1',
        '+task.pretrain_steps=1',
    ]

    def test_train(self, tmpdir):
        tmpdir = Path(tmpdir)
        result = subprocess.run(
            [*self.ARGS, f'hydra.run.dir={tmpdir}'],
            cwd=tmpdir,
            capture_output=True,
            # check=True,
        )
        files = os.listdir(tmpdir)
        print(files)
        print(result.stdout.decode())
        print(result)
        assert 'deepqmc.log' in files
        assert 'training' in files
        train_files = os.listdir(tmpdir / 'training')
        assert 'result.h5' in train_files
        assert any(f.startswith('events.out.tfevents.') for f in train_files)
        assert 'Pretraining completed' in result.stdout.decode()
        assert 'Equilibrating sampler...' in result.stdout.decode()
        assert 'Start training' in result.stdout.decode()
        assert 'The training has been completed!' in result.stdout.decode()
