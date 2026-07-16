import jax.numpy as jnp

from deepqmc.ewm import init_ewm, init_multi_mol_multi_state_ewm


class TestEwm:
    def test_update(self):
        state, update = init_ewm(max_alpha=0.9, decay_alpha=1.0, window_size=5)

        state = update(jnp.array(1.0), state)
        assert jnp.allclose(state.mean, 0.5)
        assert jnp.allclose(state.var, 0.125)
        assert jnp.allclose(state.sqerr, 0.0625)

        state = update(jnp.array(2.0), state)
        assert jnp.allclose(state.mean, 1.0)


class TestMultiMolMultiStateEwm:
    def test_update_all(self):
        state, update = init_multi_mol_multi_state_ewm(
            (2, 3), max_alpha=0.9, decay_alpha=1.0, window_size=5
        )
        state = update(jnp.ones((2, 3)), state)
        assert jnp.allclose(state.mean, 0.5)

    def test_update_sub_idxs(self):
        state, update = init_multi_mol_multi_state_ewm(
            (2, 3), max_alpha=0.9, decay_alpha=1.0, window_size=5
        )
        state = update(jnp.ones((2, 3)), state)

        state = update(2 * jnp.ones((1, 3)), state, sub_idxs=jnp.array([0]))
        assert jnp.allclose(state.mean[0], 1.0)
        assert jnp.allclose(state.mean[1], 0.5)
