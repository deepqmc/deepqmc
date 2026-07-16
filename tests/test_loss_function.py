import jax
import jax.numpy as jnp
import pytest

from deepqmc.loss.clip import (
    clip_local_energy,
    median_clip_and_mask,
    psi_ratio_clip_and_mask,
)
from deepqmc.loss.energy import (
    compute_local_energy,
    compute_mean_energy,
    compute_mean_energy_tangent,
)
from deepqmc.loss.loss_function import (
    compute_log_psi_tangent,
    create_idle_loss_fn,
    create_loss_fn,
)
from deepqmc.loss.overlap import compute_mean_overlap, compute_psi_ratio
from deepqmc.loss.spin import (
    compute_mean_spin,
    compute_spin_contributions,
    compute_spin_raising_contributions,
)
from deepqmc.parallel import pmap
from deepqmc.utils import tree_stack


def energy_clip_mask_fn(x):
    return median_clip_and_mask(x, clip_width=5.0, median_center=True)


def overlap_clip_mask_fn(x):
    return psi_ratio_clip_and_mask(x, clip_width=5.0)


def add_device_axis(tree):
    return jax.tree.map(lambda x: x[None], tree)


@pytest.fixture(scope='module')
def wf(helpers):
    hamil = helpers.hamil()
    ansatz, params = helpers.create_ansatz(hamil)
    return hamil, ansatz, params


class TestCreateIdleLossFn:
    def test_forward_is_zero(self, helpers, wf):
        hamil, ansatz, params = wf
        phys_conf, weight = helpers.batch_phys_conf(
            hamil, mol_batch=2, n_states=1, elec_batch=3
        )
        params_list_d, rng_d, batch_d = add_device_axis(
            ([params], helpers.rng(), (phys_conf, weight, None))
        )
        loss_fn = create_idle_loss_fn(hamil, ansatz)
        loss, aux = pmap(loss_fn)(params_list_d, rng_d, batch_d)
        assert jnp.allclose(loss, 0.0)
        assert aux == (None, None, {})

    def test_grad_is_zero(self, helpers, wf):
        hamil, ansatz, params = wf
        phys_conf, weight = helpers.batch_phys_conf(
            hamil, mol_batch=2, n_states=1, elec_batch=3
        )
        params_list_d, rng_d, batch_d = add_device_axis(
            ([params], helpers.rng(), (phys_conf, weight, None))
        )
        loss_fn = create_idle_loss_fn(hamil, ansatz)

        def loss_and_grad(params_list, rng, batch):
            return jax.value_and_grad(loss_fn, has_aux=True)(params_list, rng, batch)

        (loss, aux), grad = pmap(loss_and_grad)(params_list_d, rng_d, batch_d)
        assert jax.tree.structure(grad) == jax.tree.structure(params_list_d)
        assert helpers.pytree_allclose(
            grad, jax.tree.map(jnp.zeros_like, params_list_d)
        )


class TestCreateLossFnEnergyOnly:
    MOL_BATCH = 2
    ELEC_BATCH = 3

    def test_psi_ratio_none_and_loss_matches_mean_energy(self, helpers, wf):
        hamil, ansatz, params = wf
        phys_conf, weight = helpers.batch_phys_conf(
            hamil, mol_batch=self.MOL_BATCH, n_states=1, elec_batch=self.ELEC_BATCH
        )
        params_list_d, rng_d, batch_d = add_device_axis(
            ([params], helpers.rng(), (phys_conf, weight, None))
        )
        loss_fn = create_loss_fn(hamil, ansatz, energy_clip_mask_fn)
        loss, (local_energy, psi_ratio, stats) = pmap(loss_fn)(
            params_list_d, rng_d, batch_d
        )
        assert psi_ratio is None
        expected_loss, _ = pmap(compute_mean_energy)(local_energy[:, 0], weight[None])
        assert jnp.allclose(loss, expected_loss)

    def test_grad_smoke(self, helpers, wf):
        hamil, ansatz, params = wf
        phys_conf, weight = helpers.batch_phys_conf(
            hamil, mol_batch=self.MOL_BATCH, n_states=1, elec_batch=self.ELEC_BATCH
        )
        params_list_d, rng_d, batch_d = add_device_axis(
            ([params], helpers.rng(), (phys_conf, weight, None))
        )
        loss_fn = create_loss_fn(hamil, ansatz, energy_clip_mask_fn)

        def loss_and_grad(params_list, rng, batch):
            return jax.value_and_grad(loss_fn, has_aux=True)(params_list, rng, batch)

        (loss, aux), grad = pmap(loss_and_grad)(params_list_d, rng_d, batch_d)
        assert jax.tree.structure(grad) == jax.tree.structure(params_list_d)
        leaves = jax.tree.leaves(grad)
        assert all(jnp.isfinite(leaf).all() for leaf in leaves)
        assert any((leaf != 0).any() for leaf in leaves)

    def test_jvp_matches_manual_tangent(self, helpers, wf):
        hamil, ansatz, params = wf
        phys_conf, weight = helpers.batch_phys_conf(
            hamil, mol_batch=self.MOL_BATCH, n_states=1, elec_batch=self.ELEC_BATCH
        )
        params_list = [params]
        tangent_list = jax.tree.map(jnp.ones_like, params_list)
        rng = helpers.rng()
        batch = (phys_conf, weight, None)
        params_list_d, tangent_list_d, rng_d, batch_d = add_device_axis(
            (params_list, tangent_list, rng, batch)
        )
        loss_fn = create_loss_fn(hamil, ansatz, energy_clip_mask_fn)

        def run_jvp(params_list, tangent_list, rng, batch):
            return jax.jvp(
                lambda p: loss_fn(p, rng, batch), (params_list,), (tangent_list,)
            )

        (_, _), (loss_tangent, _) = pmap(run_jvp)(
            params_list_d, tangent_list_d, rng_d, batch_d
        )

        def manual_tangent(params_list, tangent_list, rng, batch):
            phys_conf, weight, _ = batch
            stacked_params = tree_stack(params_list)
            _, rng_energy = jax.random.split(rng)
            local_energy, _ = compute_local_energy(
                rng_energy, hamil, ansatz.apply, stacked_params, phys_conf, None
            )
            clipped_local_energy, gradient_mask = clip_local_energy(
                energy_clip_mask_fn, local_energy
            )
            log_psi_tangent = compute_log_psi_tangent(
                ansatz, phys_conf, params_list, tangent_list
            )
            return compute_mean_energy_tangent(
                clipped_local_energy, weight, log_psi_tangent, gradient_mask
            )

        expected_tangent = pmap(manual_tangent)(
            params_list_d, tangent_list_d, rng_d, batch_d
        )
        assert jnp.allclose(loss_tangent, expected_tangent)
        assert not jnp.allclose(loss_tangent, 0.0)


class TestCreateLossFnWithOverlap:
    MOL_BATCH = 2
    N_STATES = 2
    ELEC_BATCH = 3
    ALPHA = 1.0

    def _batch(self, helpers, hamil):
        phys_conf, weight = helpers.batch_phys_conf(
            hamil,
            mol_batch=self.MOL_BATCH,
            n_states=self.N_STATES,
            elec_batch=self.ELEC_BATCH,
        )
        data = {'energy_ewm': jnp.zeros((self.MOL_BATCH, self.N_STATES))}
        return phys_conf, weight, data

    def test_psi_ratio_present_and_overlap_stats(self, helpers, wf):
        hamil, ansatz, params = wf
        batch = self._batch(helpers, hamil)
        params_list_d, rng_d, batch_d = add_device_axis(
            ([params] * self.N_STATES, helpers.rng(), batch)
        )
        loss_fn = create_loss_fn(
            hamil,
            ansatz,
            energy_clip_mask_fn,
            clip_mask_overlap_fn=overlap_clip_mask_fn,
            alpha=self.ALPHA,
        )
        loss, (local_energy, psi_ratio, stats) = pmap(loss_fn)(
            params_list_d, rng_d, batch_d
        )
        assert psi_ratio is not None
        assert 'overlap/pairwise/mean' in stats

    def test_loss_matches_energy_plus_overlap(self, helpers, wf):
        hamil, ansatz, params = wf
        phys_conf, weight, data = self._batch(helpers, hamil)
        params_list = [params] * self.N_STATES
        params_list_d, rng_d, batch_d = add_device_axis(
            (params_list, helpers.rng(), (phys_conf, weight, data))
        )
        loss_fn = create_loss_fn(
            hamil,
            ansatz,
            energy_clip_mask_fn,
            clip_mask_overlap_fn=overlap_clip_mask_fn,
            alpha=self.ALPHA,
        )
        loss, (local_energy, psi_ratio, stats) = pmap(loss_fn)(
            params_list_d, rng_d, batch_d
        )
        energy_loss, _ = pmap(compute_mean_energy)(local_energy[:, 0], weight[None])

        stacked_params_d = add_device_axis(tree_stack(params_list))
        phys_conf_d = add_device_axis(phys_conf)
        weight_d = add_device_axis(weight)

        def overlap_loss_fn(stacked_params, phys_conf, weight):
            psi_ratio, _ = compute_psi_ratio(ansatz, stacked_params, phys_conf)
            return compute_mean_overlap(psi_ratio, weight)

        overlap_loss, _ = pmap(overlap_loss_fn)(stacked_params_d, phys_conf_d, weight_d)
        expected_loss = energy_loss + self.ALPHA * overlap_loss
        assert jnp.allclose(loss, expected_loss)

    def test_grad_smoke(self, helpers, wf):
        hamil, ansatz, params = wf
        batch = self._batch(helpers, hamil)
        params_list_d, rng_d, batch_d = add_device_axis(
            ([params] * self.N_STATES, helpers.rng(), batch)
        )
        loss_fn = create_loss_fn(
            hamil,
            ansatz,
            energy_clip_mask_fn,
            clip_mask_overlap_fn=overlap_clip_mask_fn,
            alpha=self.ALPHA,
        )

        def loss_and_grad(params_list, rng, batch):
            return jax.value_and_grad(loss_fn, has_aux=True)(params_list, rng, batch)

        (loss, aux), grad = pmap(loss_and_grad)(params_list_d, rng_d, batch_d)
        assert jax.tree.structure(grad) == jax.tree.structure(params_list_d)
        leaves = jax.tree.leaves(grad)
        assert all(jnp.isfinite(leaf).all() for leaf in leaves)
        assert any((leaf != 0).any() for leaf in leaves)


class TestCreateLossFnWithSpinPenalty:
    MOL_BATCH = 2
    ELEC_BATCH = 3
    SPIN_PENALTY = 0.1

    def _batch(self, helpers, hamil):
        phys_conf, weight = helpers.batch_phys_conf(
            hamil, mol_batch=self.MOL_BATCH, n_states=1, elec_batch=self.ELEC_BATCH
        )
        return phys_conf, weight, None

    def test_squared_matches_manual_spin(self, helpers, wf):
        hamil, ansatz, params = wf
        phys_conf, weight, data = self._batch(helpers, hamil)
        params_list = [params]
        params_list_d, rng_d, batch_d = add_device_axis(
            (params_list, helpers.rng(), (phys_conf, weight, data))
        )
        loss_fn = create_loss_fn(
            hamil,
            ansatz,
            energy_clip_mask_fn,
            spin_penalty=self.SPIN_PENALTY,
            spin_penalty_type='squared',
        )
        loss, (local_energy, psi_ratio, stats) = pmap(loss_fn)(
            params_list_d, rng_d, batch_d
        )
        energy_loss, _ = pmap(compute_mean_energy)(local_energy[:, 0], weight[None])

        stacked_params_d = add_device_axis(tree_stack(params_list))
        phys_conf_d = add_device_axis(phys_conf)
        weight_d = add_device_axis(weight)

        def spin_loss_fn(stacked_params, phys_conf, weight):
            spin_contributions = compute_spin_contributions(
                hamil, ansatz, stacked_params, phys_conf, None
            )
            return compute_mean_spin(spin_contributions, weight, None)

        spin_value, _ = pmap(spin_loss_fn)(stacked_params_d, phys_conf_d, weight_d)
        expected_loss = energy_loss + self.SPIN_PENALTY * spin_value
        assert jnp.allclose(loss, expected_loss)

    def test_raising_matches_manual_spin(self, helpers, wf):
        hamil, ansatz, params = wf
        phys_conf, weight, data = self._batch(helpers, hamil)
        params_list = [params]
        rng = helpers.rng()
        params_list_d, rng_d, batch_d = add_device_axis(
            (params_list, rng, (phys_conf, weight, data))
        )
        loss_fn = create_loss_fn(
            hamil,
            ansatz,
            energy_clip_mask_fn,
            spin_penalty=self.SPIN_PENALTY,
            spin_penalty_type='raising',
        )
        loss, _ = pmap(loss_fn)(params_list_d, rng_d, batch_d)

        stacked_params_d = add_device_axis(tree_stack(params_list))
        phys_conf_d = add_device_axis(phys_conf)
        weight_d = add_device_axis(weight)

        def manual(rng, stacked_params, phys_conf, weight):
            rng, rng_energy = jax.random.split(rng)
            local_energy, _ = compute_local_energy(
                rng_energy, hamil, ansatz.apply, stacked_params, phys_conf, None
            )
            energy_loss, _ = compute_mean_energy(local_energy, weight)
            _, rng_spin_raising = jax.random.split(rng)
            spin_contributions = compute_spin_raising_contributions(
                rng_spin_raising, hamil, ansatz, phys_conf, stacked_params, None, None
            )
            spin_value, _ = compute_mean_spin(spin_contributions, weight, None)
            return energy_loss + self.SPIN_PENALTY * spin_value

        expected_loss = pmap(manual)(rng_d, stacked_params_d, phys_conf_d, weight_d)
        assert jnp.allclose(loss, expected_loss)

    def test_grad_smoke_squared(self, helpers, wf):
        hamil, ansatz, params = wf
        batch = self._batch(helpers, hamil)
        params_list_d, rng_d, batch_d = add_device_axis(
            ([params], helpers.rng(), batch)
        )
        loss_fn = create_loss_fn(
            hamil,
            ansatz,
            energy_clip_mask_fn,
            spin_penalty=self.SPIN_PENALTY,
            spin_penalty_type='squared',
        )

        def loss_and_grad(params_list, rng, batch):
            return jax.value_and_grad(loss_fn, has_aux=True)(params_list, rng, batch)

        (loss, aux), grad = pmap(loss_and_grad)(params_list_d, rng_d, batch_d)
        assert jax.tree.structure(grad) == jax.tree.structure(params_list_d)
        leaves = jax.tree.leaves(grad)
        assert all(jnp.isfinite(leaf).all() for leaf in leaves)
        assert any((leaf != 0).any() for leaf in leaves)

    def test_grad_smoke_raising(self, helpers, wf):
        hamil, ansatz, params = wf
        batch = self._batch(helpers, hamil)
        params_list_d, rng_d, batch_d = add_device_axis(
            ([params], helpers.rng(), batch)
        )
        loss_fn = create_loss_fn(
            hamil,
            ansatz,
            energy_clip_mask_fn,
            spin_penalty=self.SPIN_PENALTY,
            spin_penalty_type='raising',
        )

        def loss_and_grad(params_list, rng, batch):
            return jax.value_and_grad(loss_fn, has_aux=True)(params_list, rng, batch)

        (loss, aux), grad = pmap(loss_and_grad)(params_list_d, rng_d, batch_d)
        assert jax.tree.structure(grad) == jax.tree.structure(params_list_d)
        leaves = jax.tree.leaves(grad)
        assert all(jnp.isfinite(leaf).all() for leaf in leaves)
        assert any((leaf != 0).any() for leaf in leaves)


class TestLocalEnergyBatchSizeConsistency:
    MOL_BATCH = 2
    ELEC_BATCH = 4

    def test_forward_loss_matches_across_batch_sizes(self, helpers, wf):
        hamil, ansatz, params = wf
        phys_conf, weight = helpers.batch_phys_conf(
            hamil, mol_batch=self.MOL_BATCH, n_states=1, elec_batch=self.ELEC_BATCH
        )
        params_list_d, rng_d, batch_d = add_device_axis(
            ([params], helpers.rng(), (phys_conf, weight, None))
        )
        loss_fn_none = create_loss_fn(
            hamil, ansatz, energy_clip_mask_fn, local_energy_batch_size=None
        )
        loss_fn_batched = create_loss_fn(
            hamil, ansatz, energy_clip_mask_fn, local_energy_batch_size=2
        )
        loss_none, (local_energy_none, _, _) = pmap(loss_fn_none)(
            params_list_d, rng_d, batch_d
        )
        loss_batched, (local_energy_batched, _, _) = pmap(loss_fn_batched)(
            params_list_d, rng_d, batch_d
        )
        assert jnp.allclose(loss_none, loss_batched)
        assert jnp.allclose(local_energy_none, local_energy_batched)
