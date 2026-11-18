from types import SimpleNamespace

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

import deepqmc.train as train_module
from deepqmc.log import H5Logger
from deepqmc.types import TrainState


class TestH5Logger:
    def write_steps(self, logger, steps, offset=0.0):
        for step in steps:
            logger.update({'local_energy': {'samples': np.full(2, step + offset)}})

    def test_rewind(self, tmp_path):
        logger = H5Logger(str(tmp_path), 0)
        self.write_steps(logger, range(10))
        logger.rewind(5)
        assert logger.h5file['local_energy/samples'].shape == (5, 2)
        self.write_steps(logger, range(5, 10), offset=0.5)
        samples = logger.h5file['local_energy/samples'][:, 0]
        np.testing.assert_array_equal(samples, [0, 1, 2, 3, 4, 5.5, 6.5, 7.5, 8.5, 9.5])
        logger.close()

    def test_rewind_in_new_workdir_of_restarted_run(self, tmp_path):
        # a run restarted from step 100 logs into a fresh result.h5,
        # whose first row corresponds to step 100
        logger = H5Logger(str(tmp_path), 100)
        self.write_steps(logger, range(100, 110))
        logger.rewind(105)
        self.write_steps(logger, range(105, 110), offset=0.5)
        samples = logger.h5file['local_energy/samples'][:, 0]
        np.testing.assert_array_equal(
            samples, [100, 101, 102, 103, 104, 105.5, 106.5, 107.5, 108.5, 109.5]
        )
        logger.close()

    def test_rewind_in_existing_result_file(self, tmp_path):
        # reopening an existing result.h5 at step 5 truncates it to 5 rows,
        # rows stay indexed by absolute step
        logger = H5Logger(str(tmp_path), 0)
        self.write_steps(logger, range(10))
        logger.close()
        logger = H5Logger(str(tmp_path), 5)
        self.write_steps(logger, range(5, 10), offset=0.25)
        logger.rewind(7)
        self.write_steps(logger, range(7, 10), offset=0.5)
        samples = logger.h5file['local_energy/samples'][:, 0]
        np.testing.assert_array_equal(
            samples, [0, 1, 2, 3, 4, 5.25, 6.25, 7.5, 8.5, 9.5]
        )
        logger.close()


class TestH5LoggerFirstStep:
    def write_steps(self, logger, steps, offset=0.0):
        for step in steps:
            row = {'local_energy': {'mean': step + offset}}
            if step % 10 == 0:
                row['spin'] = {'mean': step + offset}
            logger.update(row)

    def test_first_step_attribute(self, tmp_path):
        logger = H5Logger(str(tmp_path), 100)
        self.write_steps(logger, range(100, 103))
        assert logger.h5file.attrs['first_step'] == 100
        logger.close()
        # reopening keeps the step of the first row
        logger = H5Logger(str(tmp_path), 101)
        assert logger.h5file.attrs['first_step'] == 100
        logger.close()

    def test_reopen_file_started_at_later_step(self, tmp_path):
        logger = H5Logger(str(tmp_path), 45, ['spin'], periods={'spin': 10})
        self.write_steps(logger, range(45, 77))
        logger.close()
        logger = H5Logger(str(tmp_path), 65, ['spin'], periods={'spin': 10})
        assert logger.h5file['local_energy/mean'].shape == (20,)
        assert logger.h5file['spin/mean'].shape == (2,)
        self.write_steps(logger, range(65, 80), offset=0.5)
        logger.rewind(72)
        self.write_steps(logger, range(72, 80), offset=0.75)
        energies = logger.h5file['local_energy/mean'][:]
        np.testing.assert_array_equal(energies[:20], np.arange(45, 65))
        np.testing.assert_array_equal(energies[20:27], np.arange(65, 72) + 0.5)
        np.testing.assert_array_equal(energies[27:], np.arange(72, 80) + 0.75)
        np.testing.assert_array_equal(logger.h5file['spin/mean'][:], [50, 60, 70.5])
        logger.close()

    def test_old_file_without_attribute_starts_at_zero(self, tmp_path):
        logger = H5Logger(str(tmp_path), 0)
        self.write_steps(logger, range(10))
        del logger.h5file.attrs['first_step']
        logger.close()
        logger = H5Logger(str(tmp_path), 5)
        assert logger.h5file['local_energy/mean'].shape == (5,)
        logger.close()

    def test_reopen_before_first_step_fails(self, tmp_path):
        logger = H5Logger(str(tmp_path), 100)
        self.write_steps(logger, range(100, 103))
        logger.close()
        with pytest.raises(AssertionError):
            H5Logger(str(tmp_path), 50)


class TestH5LoggerPeriodicEntries:
    def write_steps(self, logger, steps, offset=0.0):
        for step in steps:
            row = {'local_energy': {'mean': step + offset}}
            if step % 10 == 0:
                row['spin'] = {'mean': step + offset}
            logger.update(row)

    def test_period_attribute(self, tmp_path):
        logger = H5Logger(str(tmp_path), 0, ['spin'], periods={'spin': 10})
        self.write_steps(logger, range(3))
        assert logger.h5file['spin/mean'].attrs['period'] == 10
        assert 'period' not in logger.h5file['local_energy/mean'].attrs
        logger.close()

    def test_rewind(self, tmp_path):
        logger = H5Logger(str(tmp_path), 0, ['spin'], periods={'spin': 10})
        self.write_steps(logger, range(57))
        logger.rewind(50)
        assert logger.h5file['local_energy/mean'].shape == (50,)
        assert logger.h5file['spin/mean'].shape == (5,)
        self.write_steps(logger, range(50, 70), offset=0.5)
        np.testing.assert_array_equal(
            logger.h5file['spin/mean'][:], [0, 10, 20, 30, 40, 50.5, 60.5]
        )
        assert logger.h5file['local_energy/mean'].shape == (70,)
        logger.close()

    def test_rewind_to_multiple_of_period(self, tmp_path):
        logger = H5Logger(str(tmp_path), 0, ['spin'], periods={'spin': 10})
        self.write_steps(logger, range(57))
        logger.rewind(51)
        assert logger.h5file['spin/mean'].shape == (6,)
        logger.close()

    def test_reopen_existing_result_file(self, tmp_path):
        logger = H5Logger(str(tmp_path), 0, ['spin'], periods={'spin': 10})
        self.write_steps(logger, range(57))
        logger.close()
        logger = H5Logger(str(tmp_path), 45, ['spin'], periods={'spin': 10})
        assert logger.h5file['spin/mean'].shape == (5,)
        self.write_steps(logger, range(45, 70), offset=0.5)
        np.testing.assert_array_equal(
            logger.h5file['spin/mean'][:], [0, 10, 20, 30, 40, 50.5, 60.5]
        )
        logger.close()

    def test_rewind_in_new_workdir_of_restarted_run(self, tmp_path):
        # the first row of a new result.h5 is at step 45, spin is first logged at 50
        logger = H5Logger(str(tmp_path), 45, ['spin'], periods={'spin': 10})
        self.write_steps(logger, range(45, 77))
        logger.rewind(65)
        assert logger.h5file['local_energy/mean'].shape == (20,)
        np.testing.assert_array_equal(logger.h5file['spin/mean'][:], [50, 60])
        logger.close()


class InMemoryCheckpointStore:
    def __init__(self, workdir, interval=3):
        self.interval = interval
        self.chkpts = []

    def update(self, step, state):
        if not self.chkpts or step >= self.interval + self.chkpts[-1][0]:
            self.chkpts.append((step, state))

    @property
    def last(self):
        return self.chkpts[-1]

    def close(self):
        pass


def test_train_nan_rewind_does_not_duplicate_h5_rows(tmp_path, monkeypatch):
    steps, nan_step = 10, 6
    attempts = []

    def fake_fit_wf(rng, hamil, ansatz, opt, mol_idx_sampler, sampler, pbar, *_, **__):
        attempt = len(attempts)
        attempts.append(attempt)
        for step in pbar:
            log = float(step) + 0.1 * attempt
            if attempt == 0 and step == nan_step:
                log = jnp.nan
            train_state = TrainState({'elec': None}, {}, None)
            samples = {'psi/samples': {'log': jnp.array([log])}}
            yield step, train_state, jnp.array([0]), {}, samples

    monkeypatch.setattr(train_module, 'fit_wf', fake_fit_wf)
    train_module.train(
        hamil=SimpleNamespace(mol=SimpleNamespace(coords=np.zeros((1, 3)))),
        ansatz=SimpleNamespace(apply=None),
        opt=object(),
        sampler_factory=lambda *_: (None, None),
        steps=steps,
        seed=0,
        electron_batch_size=1,
        workdir=str(tmp_path),
        train_state=TrainState({'elec': None}, {}, None),
        chkpt_constructor=InMemoryCheckpointStore,
    )

    assert len(attempts) == 2
    with h5py.File(tmp_path / 'training/result.h5') as f:
        log = f['psi/samples/log'][:, 0]
    # the first attempt logs steps 0..5 and checkpoints chkpt-1 and chkpt-4
    # (chkpt-i holds the state before step i), so the NaN at step 6 rewinds to
    # step 4 and steps 4..9 are replayed by the second attempt
    np.testing.assert_allclose(log, [0, 1, 2, 3, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1])
