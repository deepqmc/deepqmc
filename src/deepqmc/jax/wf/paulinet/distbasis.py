import jax.numpy as jnp

__version__ = '0.1.0'
__all__ = ['DistanceBasis']


class DistanceBasis:
    r"""Expands distances in distance feature basis.

    Maps distances, *d*, to distance features, :math:`\mathbf e(d)`.

    Args:
        dist_feat_dim (int): :math:`\dim(\mathbf e)`, number of features
        cutoff (float, .a.u.): distance at which all features go to zero
        envelope (str): type of envelope for distance features

            - ``'physnet'``: taken from [UnkeJCTC19]_
            - ``'nocusp'``: used in [HermannNC20]_

    Shape:
        - Input, *d*: :math:`(*)`
        - Output, :math:`\mathbf e(d)`: :math:`(*,\dim(\mathbf e))`
    """

    def __init__(
        self,
        dist_feat_dim,
        cutoff=10.0,
        envelope='physnet',
        smooth=None,
        offset=True,
        powers=None,
        eps=1e-2,
    ):
        super().__init__()
        if powers:
            self.powers = jnp.array(powers)
            self.eps = eps
            dist_feat_dim -= len(powers)
        else:
            self.powers = None
        delta = 1 / (2 * dist_feat_dim) if offset else 0
        qs = jnp.linspace(delta, 1 - delta, dist_feat_dim)
        self.cutoff = cutoff
        self.envelope = envelope
        self.mus = cutoff * qs**2
        self.sigmas = (1 + cutoff * qs) / 7
        self.smooth = smooth

    def __call__(self, dists):
        if self.smooth is not None:
            dists = dists + 1 / self.smooth * jnp.log(
                1 + jnp.exp(-2 * self.smooth * dists)
            )
        if self.envelope == 'physnet':
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
        x = envelope[..., None] * jnp.exp(
            -((dists[..., None] - self.mus) ** 2) / self.sigmas**2
        )
        if self.powers is not None:
            powers = jnp.where(
                self.powers > 0,
                dists[..., None] ** self.powers,
                1 / (dists[..., None] ** (-self.powers) + self.eps),
            )
            x = jnp.concatenate([powers, x], axis=-1)
        return x

    def extra_repr(self):
        return ', '.join(
            f'{lbl}={val!r}'
            for lbl, val in [
                ('dist_feat_dim', len(self.mus)),
                ('cutoff', self.cutoff),
                ('envelope', self.envelope),
            ]
        )
