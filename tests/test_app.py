import os
import subprocess
from pathlib import Path


class TestApp:
    ARGS = ['deepqmc', 'device=cpu', 'task.steps=0', '+task.max_eq_steps=0']

    def test_train(self, tmpdir):
        tmpdir = Path(tmpdir)
        result = subprocess.run(
            [*self.ARGS, f'hydra.run.dir={tmpdir}'],
            cwd=tmpdir,
            capture_output=True,
            check=True,
        )
        files = os.listdir(tmpdir)
        assert 'deepqmc.log' in files
        assert 'train' in files
        train_files = os.listdir(tmpdir / 'train')
        assert 'result.h5' in train_files
        assert any(f.startswith('events.out.tfevents.') for f in train_files)
        assert 'Start train' in result.stdout.decode()
