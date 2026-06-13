import os
import socket
import subprocess
from pathlib import Path


def free_port() -> int:
    r"""Return a currently free TCP port on the loopback interface."""
    with socket.socket() as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]


def run_multihost(
    args: list[str], rundir: Path, n_proc: int = 2, timeout: int = 600
) -> tuple[list[int], list[str]]:
    r"""Launch ``args`` as ``n_proc`` JAX processes sharing ``rundir``.

    Simulates a multi-host run on a single machine: each process is given a single
    forced CPU device and the deepqmc multi-process environment variables, so they
    coordinate through ``jax.distributed``. Each process writes its merged
    stdout/stderr to ``rundir/proc_{i}.out``.

    Returns the per-process return codes and captured outputs.
    """
    port = free_port()
    procs, logfiles = [], []
    for process_id in range(n_proc):
        env = {
            **os.environ,
            'JAX_PLATFORMS': 'cpu',
            'XLA_FLAGS': '--xla_force_host_platform_device_count=1',
            'JAX_COORDINATOR_ADDRESS': f'localhost:{port}',
            'DEEPQMC_NUM_PROCESSES': str(n_proc),
            'DEEPQMC_PROCESS_ID': str(process_id),
        }
        logfile = open(rundir / f'proc_{process_id}.out', 'w')
        logfiles.append(logfile)
        procs.append(
            subprocess.Popen(
                [*args, f'hydra.run.dir={rundir}'],
                cwd=rundir,
                env=env,
                stdout=logfile,
                stderr=subprocess.STDOUT,
            )
        )
    try:
        returncodes = [proc.wait(timeout=timeout) for proc in procs]
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
        for logfile in logfiles:
            logfile.close()
    outputs = [(rundir / f'proc_{i}.out').read_text() for i in range(n_proc)]
    return returncodes, outputs


class TestAppMultihost:
    # multi-host run with two processes (one CPU device each) and a checkpoint
    # written every step, so that a restart has a checkpoint to load
    ARGS = [
        'deepqmc',
        'hamil/mol=H2',
        'task.steps=2',
        'task.electron_batch_size=2',
        '+task.max_eq_steps=1',
        'task.pretrain_steps=1',
        '+task.chkpt_constructor.interval=1',
    ]

    def test_train_and_restart(self, tmpdir):
        tmpdir = Path(tmpdir)

        # multi-host training across two processes
        returncodes, outputs = run_multihost(self.ARGS, tmpdir)
        assert returncodes == [0, 0], '\n'.join(outputs)
        for process_id in range(2):
            train_dir = tmpdir / f'training_{process_id}'
            assert train_dir.is_dir()
            train_files = os.listdir(train_dir)
            assert 'result.h5' in train_files
            assert any(f.startswith('chkpt-') for f in train_files)
            assert 'The training has been completed!' in outputs[process_id]

        # restart both processes from their per-process checkpoints. The
        # process_idx_suffix resolver expands to training_{i} in each process.
        restart_dir = tmpdir / 'restart'
        restart_dir.mkdir()
        restart_args = [
            'deepqmc',
            'task=restart',
            'task.restdir=' + str(tmpdir / 'training') + '${process_idx_suffix:}',
            '+task.steps=4',
        ]
        returncodes, outputs = run_multihost(restart_args, restart_dir)
        assert returncodes == [0, 0], '\n'.join(outputs)
        for process_id in range(2):
            assert (restart_dir / f'training_{process_id}').is_dir()
            assert 'Restart training from step' in outputs[process_id]
            assert 'The training has been completed!' in outputs[process_id]
