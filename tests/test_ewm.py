import jax.numpy as jnp
import pytest

from deepqmc.ewm import init_ewm, init_multi_mol_multi_state_ewm


class TestEwm:
    def test_update(self):
        state, update = init_ewm(max_alpha=0.9, decay_alpha=1.0, window_size=5)

        state = update(jnp.array(1.0), state)
        assert jnp.allclose(state.mean, 1.0)
        assert jnp.isinf(state.var)
        assert jnp.isinf(state.sqerr)

        # alpha = 1 / 3, weights = [1 / 3, 2 / 3] on the buffer [2, 1]
        state = update(jnp.array(2.0), state)
        assert jnp.allclose(state.mean, 4 / 3)
        assert jnp.allclose(state.var, 2 / 9)
        assert jnp.allclose(state.sqerr, 8 / 81)

    @pytest.mark.parametrize('window_size', [5, None])
    def test_constant_input_is_unbiased(self, window_size):
        # covers both a partially filled and (for window_size=5) a full window
        state, update = init_ewm(decay_alpha=1.0, window_size=window_size)
        for step in range(20):
            state = update(jnp.array(10.0), state)
            assert jnp.allclose(state.mean, 10.0), step
            if step > 0:
                assert jnp.allclose(state.var, 0.0, atol=1e-10), step


class TestMultiMolMultiStateEwm:
    def test_update_all(self):
        state, update = init_multi_mol_multi_state_ewm(
            (2, 3), max_alpha=0.9, decay_alpha=1.0, window_size=5
        )
        state = update(jnp.ones((2, 3)), state)
        assert jnp.allclose(state.mean, 1.0)

    def test_update_sub_idxs(self):
        state, update = init_multi_mol_multi_state_ewm(
            (2, 3), max_alpha=0.9, decay_alpha=1.0, window_size=5
        )
        state = update(jnp.ones((2, 3)), state)

        state = update(2 * jnp.ones((1, 3)), state, sub_idxs=jnp.array([0]))
        assert jnp.allclose(state.mean[0], 4 / 3)
        assert jnp.allclose(state.mean[1], 1.0)

    def test_first_update_per_molecule(self):
        # molecule 1 is first updated at the second call, it must not be biased
        state, update = init_multi_mol_multi_state_ewm(
            (2, 1), max_alpha=0.9, decay_alpha=1.0, window_size=5
        )
        state = update(jnp.ones((1, 1)), state, sub_idxs=jnp.array([0]))
        state = update(5 * jnp.ones((1, 1)), state, sub_idxs=jnp.array([1]))
        assert jnp.allclose(state.mean[1], 5.0)
        assert jnp.isinf(state.sqerr[1]).all()
