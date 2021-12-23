import os
import subprocess

ARGS = ['deepqmc.app', 'device=cpu', 'system=H2', 'hydra.run.dir=.']


def test_train(tmpdir):
    result = subprocess.run(
        [*ARGS, 'task.n_steps=0', 'task.equilibrate=false'],
        cwd=tmpdir,
        capture_output=True,
        check=True,
    )
    files = os.listdir(tmpdir)
    assert 'fit.h5' in files
    assert 'pyscf.chk' in files
    assert 'chkpts' in files
    assert any(f.startswith('events.out.tfevents.') for f in files)
    assert 'Initializing training' in result.stdout.decode()


def test_evaluate(tmpdir):
    result = subprocess.run(
        [
            *ARGS,
            'task=evaluate',
            'fromdir=null',
            'state=null',
            'ansatz=paulinet',
            'task.n_steps=0',
            'task.sample_kwargs.equilibrate=false',
        ],
        cwd=tmpdir,
        capture_output=True,
        check=True,
    )
    files = os.listdir(tmpdir)
    assert 'sample.h5' in files
    assert 'pyscf.chk' in files
    assert any(f.startswith('events.out.tfevents.') for f in files)
    assert 'Moved to cpu' in result.stdout.decode()
