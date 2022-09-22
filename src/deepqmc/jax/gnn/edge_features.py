import jax.numpy as jnp

from ..utils import norm

__version__ = '0.1.0'
__all__ = ['EdgeFeatures']


class EdgeFeatures:
    r"""Computes edge features for message passing in GNNs.

    Maps particle position difference vectors, *r*, to edge features,
    :math:`\mathbf e(r)`.

    Args:
        feature_dim (int): :math:`\dim(\mathbf e)`, number of features
        cutoff (float, a.u.): distance at which all features go to zero
        envelope (str): optional, the type of envelope the distance
            features are multiplied with

            - ``'physnet'``: taken from [UnkeJCTC19]_
            - ``'nocusp'``: used in [HermannNC20]_
            - ``'nocusp_smooth_cutoff'``: a modification of ``'nocusp'``
                 where the value and first derivative of the distance
                 features are zero at the cutoff distance
        smooth (float, a.u.): if specified, distances are smoothed
            around zero such that they have a vanishing first derivative.
        offset (bool): if ``True`` the mean of the first distance feature
            is shifted away from zero.
        powers (List[int]): if specified, the first ``len(powers)`` edge
            features will be :math:`\vert \mathbf r \vert ^\text{powers}`.
        eps (float): numerical epsilon applied in the computation of
            ``powers``<0.
        difference (bool): If ``True`,` the last 3 edge features will be
            the plain difference vectors.
        safe (bool): If ``True``, a numerical epsilon is added to distances
            around zero to avoid instabilities of ```jnp.linalg.norm```
            tangents.

    Shape:
        - Input, *r*: :math:`(*,3)`
        - Output, :math:`\mathbf e(r)`: :math:`(*,\dim(\mathbf e))`
    """

    def __init__(
        self,
        feature_dim,
        cutoff=10.0,
        envelope=None,
        smooth=None,
        offset=True,
        powers=None,
        eps=1e-2,
        difference=False,
        safe=False,
    ):
        self.feature_dim = feature_dim
        dist_feat_dim = feature_dim
        if powers:
            self.powers = jnp.array(powers)
            self.eps = eps
            dist_feat_dim -= len(powers)
            assert dist_feat_dim >= 0
        else:
            self.powers = None
        if difference:
            self.difference = True
            dist_feat_dim -= 3
            assert dist_feat_dim >= 0
        else:
            self.difference = False
        self.smooth = smooth
        if dist_feat_dim > 0:
            delta = 1 / (2 * dist_feat_dim) if offset else 0
            qs = jnp.linspace(delta, 1 - delta, dist_feat_dim)
            self.cutoff = cutoff
            self.envelope = envelope
            self.mus = cutoff * qs**2
            self.sigmas = (1 + cutoff * qs) / 7
        elif powers is not None or difference:
            self.envelope = None
            self.mus = None
        else:
            raise AssertionError()
        self.safe = safe

    def __call__(self, rs):
        dists = norm(rs, self.safe)
        if self.smooth is not None:
            dists = dists + 1 / self.smooth * jnp.log(
                1 + jnp.exp(-2 * self.smooth * dists)
            )
        if self.envelope is None:
            envelope = jnp.ones_like(dists)
        elif self.envelope == 'physnet':
            dists_rel = dists / self.cutoff
            envelope = jnp.where(
                dists_rel > 1,
                jnp.zeros_like(dists, shape=(1,)),
                1 - 6 * dists_rel**5 + 15 * dists_rel**4 - 10 * dists_rel**3,
            )
        elif self.envelope == 'nocusp':
            envelope = dists**2 * jnp.exp(-dists)
        elif self.envelope == 'nocusp_smooth_cutoff':
            f = (
                lambda x: (x * (self.cutoff - x)) ** 2
                * jnp.exp(-x / 3)
                * (x < self.cutoff)
            )
            f_max = f(0.5 * (-jnp.sqrt(self.cutoff**2 + 144) + self.cutoff + 12))
            envelope = f(dists) / f_max
        else:
            raise AssertionError()
        x = (
            envelope[..., None]
            * jnp.exp(-((dists[..., None] - self.mus) ** 2) / self.sigmas**2)
            if self.mus is not None
            else jnp.zeros_like(dists, shape=(*dists.shape, 0))
        )
        if self.powers is not None:
            powers = jnp.where(
                self.powers > 0,
                dists[..., None] ** self.powers,
                1 / (dists[..., None] ** (-self.powers) + self.eps),
            )
            x = jnp.concatenate([powers, x], axis=-1)
        if self.difference:
            x = jnp.concatenate([x, rs], axis=-1)
        return x

    def extra_repr(self):
        return 'EdgeFeatures: ' + ', '.join(
            f'{lbl}={val!r}'
            for lbl, val in [
                ('feature_dim', self.feature_dim),
                ('cutoff', self.cutoff),
                ('envelope', self.envelope),
                ('smooth', self.smooth),
                ('offset', self.offset),
                ('powers', self.powers),
                ('eps', self.eps),
                ('difference', self.difference),
                ('safe', self.safe),
            ]
        )
