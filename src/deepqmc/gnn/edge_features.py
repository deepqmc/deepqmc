import jax.numpy as jnp

from ..utils import norm


class EdgeFeature:
    r"""Base class for all edge features."""

    def __len__(self):
        """Return the length of the output feature vector."""
        raise NotImplementedError


class DifferenceEdgeFeature(EdgeFeature):
    """Return the difference vector as the edge features."""

    def __init__(self, *, log_rescale=False):
        self.log_rescale = log_rescale

    def __call__(self, d):
        if self.log_rescale:
            r = norm(d, safe=True)
            d *= (jnp.log1p(r) / r)[..., None]
        return d

    def __len__(self):
        return 3


class DistancePowerEdgeFeature(EdgeFeature):
    """Return powers of the distance as edge features."""

    def __init__(self, *, powers, eps=None, log_rescale=False):
        if any(p < 0 for p in powers):
            assert eps is not None
        self.powers = jnp.asarray(powers)
        self.eps = eps or 0.0
        self.log_rescale = log_rescale

    def __call__(self, d):
        r = norm(d, safe=True)
        powers = jnp.where(
            self.powers > 0,
            r[..., None] ** self.powers,
            1 / (r[..., None] ** (-self.powers) + self.eps),
        )
        if self.log_rescale:
            powers *= (jnp.log1p(r) / r)[..., None]
        return powers

    def __len__(self):
        return len(self.powers)


class GaussianEdgeFeature(EdgeFeature):
    r"""
    Expand the distance in a Gaussian basis.

    Args:
        n_gaussian (int): the number of gaussians to use,
            consequently the length of the feature vector
        radius (float): the radius within which to place gaussians
        offset (bool): whether to offset the position of the first
            Gaussian from zero.
    """

    def __init__(self, *, n_gaussian, radius, offset):
        delta = 1 / (2 * n_gaussian) if offset else 0
        qs = jnp.linspace(delta, 1 - delta, n_gaussian)
        self.mus = radius * qs**2
        self.sigmas = (1 + radius * qs) / 7

    def __call__(self, d):
        r = norm(d, safe=True)
        gaussians = jnp.exp(-((r[..., None] - self.mus) ** 2) / self.sigmas**2)
        return gaussians

    def __len__(self):
        return len(self.mus)


class CombinedEdgeFeature(EdgeFeature):
    r"""Combine multiple edge features.

    Args:
        features (Sequence): a :data:`Sequence` of edge feature objects
            to combine.
    """

    def __init__(self, *, features):
        self.features = features

    def __call__(self, d):
        return jnp.concatenate([f(d) for f in self.features], axis=-1)

    def __len__(self):
        return sum(map(len, self.features))
