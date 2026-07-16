import jax.numpy as jnp

from deepqmc.geom import angle, distance, pairwise_distance, pairwise_self_distance
from deepqmc.geom.coordinate_transform import ZMatrixCoordinateTransform
from deepqmc.geom.zmatrix import ConcreteZMatrixTemplate


class TestPairwiseHelpers:
    def test_pairwise_self_distance(self):
        coords = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
        assert jnp.allclose(pairwise_self_distance(coords), jnp.array([1.4]))

    def test_pairwise_distance(self):
        r = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        R = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
        expected = jnp.array([[0.0, 1.4], [1.0, 0.4]])
        assert jnp.allclose(pairwise_distance(r, R), expected)


class TestInternalCoordinates:
    def test_right_angle(self):
        coords = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert jnp.allclose(distance(0, 1, coords), 1.0)
        assert jnp.allclose(angle(0, 1, 2, coords), jnp.pi / 2)


class TestZMatrixCoordinateTransformRoundTrip:
    def test_from_cartesian_of_to_cartesian_is_identity(self):
        template = ConcreteZMatrixTemplate.from_simplified_config(
            [[None, None, None], [0, None, None], [0, 1, None], [0, 1, 2]]
        )
        transform = ZMatrixCoordinateTransform(template)
        values = jnp.array([1.0, 1.2, jnp.pi / 2, 1.3, jnp.pi / 3, jnp.pi / 4])
        assert len(transform) == len(values)

        cartesian = transform.to_cartesian(values)
        recovered = transform.from_cartesian(cartesian)
        assert jnp.allclose(recovered, values, atol=1e-5)
