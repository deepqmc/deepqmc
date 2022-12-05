from collections.abc import Sequence
from typing import Optional

import e3nn_jax as e3nn
import jax.numpy as jnp
from jax.tree_util import tree_reduce


class Envelope:
    r"""
    Base class for the radial envelopes.

    Args:
        r_cut (float): the cutoff distance.
    """

    def __init__(self, r_cut):
        self.r_cut = r_cut

    def __call__(self, r):
        raise NotImplementedError


class IdentityEnvelope(Envelope):
    r"""
    The identity envelope.

    The value of the envelope is one below the cutoff distance and zero above it.

    Args:
        r_cut (float): the cutoff distance.
    """

    def __call__(self, r):
        return jnp.where(r > self.r_cut, jnp.zeros_like(r), jnp.ones_like(r))


class PolynomialEnvelope(Envelope):
    r"""
    A polynomial envelope with custom derivative properties at zero and the cutoff.

    Args:
        r_cut (float): the cutoff distance.
        n0 (int): the first :data:`n0` derivatives will be zero at the origin.
        n1 (int): the first :data:`n1` derivatives will be zero at the cutoff.
    """

    def __init__(self, r_cut: float, n0: int, n1: int):
        super().__init__(r_cut)
        self.envelope = e3nn.poly_envelope(n0, n1, r_cut)

    def __call__(self, r):
        return self.envelope(r)


class RadialBasis:
    r"""
    Base class for the radial bases.

    Args:
        n_rbf (int): the number of basis functions.
        r_cut (float): the cutoff radius for the envelope.
        envelope_factory (Callable): optional, creates the distance envelope.
        kwargs: (dict): optional, kwargs for the envelope function.
    """

    def __init__(
        self,
        n_rbf,
        r_cut,
        *,
        envelope_factory: Optional = None,
        envelope_kwargs: Optional[dict] = None,
    ):
        self.n_rbf = n_rbf
        if envelope_factory is None:
            envelope_factory = IdentityEnvelope
        self.envelope = envelope_factory(r_cut, **(envelope_kwargs or {}))

    def __call__(self, r):
        raise NotImplementedError

    def apply_envelope(self, r, rbfs):
        envelope = self.envelope(r)[..., None]
        return envelope * rbfs


class BesselBasis(RadialBasis):
    r"""
    Bessel function radial basis.

    Args:
        n_rbf (int): the number of basis functions.
        r_cut (float): the cutoff radius for the envelope.
        envelope_factory (Callable): optional, creates the distance envelope.
        kwargs: (dict): optional, kwargs for the envelope function.
    """

    def __call__(self, r):
        return self.apply_envelope(
            r,
            e3nn.bessel(r, self.n_rbf, self.envelope.r_cut),
        )


class DistancePowersBasis(RadialBasis):
    r"""A distance basis composed of integer powers of the distance.

    Args:
        n_rbf (int): the number of Gaussian basis functions.
        r_cut (float): the cutoff radius for the radial bases.
        powers (List[int]): the powers to which the distance is raised.
        eps (float): default 0.01, the numerical epsilon used for negative powers.
        envelope_factory (Callable): optional, a function returning an instance of
            :class:`~deepqmc.gnn.edge_features.Envelope`.
        envelope_kwargs: (dict): optional, kwargs for the envelope factory.
    """

    def __init__(
        self,
        n_rbf,
        r_cut,
        powers: Sequence[int],
        eps: float = 1.0e-2,
        *,
        envelope_factory: Optional = None,
        envelope_kwargs: Optional[dict] = None,
    ):
        if n_rbf != len(powers):
            raise ValueError(
                f'Length of powers ({len(powers)}) is not equal to n_rbf ({n_rbf})'
            )
        super().__init__(
            n_rbf,
            r_cut,
            envelope_factory=envelope_factory,
            envelope_kwargs=envelope_kwargs,
        )
        self.powers = jnp.asarray(powers)
        self.eps = eps

    def __call__(self, r):
        powers = jnp.where(
            self.powers > 0,
            r[..., None] ** self.powers,
            1 / (r[..., None] ** (-self.powers) + self.eps),
        )
        return self.apply_envelope(r, powers)


class GaussianBasis(RadialBasis):
    r"""Gaussian distance basis.

    Args:
        n_rbf (int): the number of Gaussian basis functions.
        r_cut (float): the cutoff radius for the radial bases.
        offset (bool): default :data:`False`, whether to offset the first Gaussian
            from zero.
        envelope_factory (Callable): optional, a function returning an instance of
            :class:`~deepqmc.gnn.edge_features.Envelope`.
        envelope_kwargs: (dict): optional, kwargs for the envelope factory.
    """

    def __init__(
        self,
        n_rbf,
        r_cut,
        offset: bool = False,
        *,
        envelope_factory: Optional = None,
        envelope_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            n_rbf,
            r_cut,
            envelope_factory=envelope_factory,
            envelope_kwargs=envelope_kwargs,
        )
        delta = 1 / (2 * n_rbf) if offset else 0
        qs = jnp.linspace(delta, 1 - delta, n_rbf)
        self.mus = r_cut * qs**2
        self.sigmas = (1 + r_cut * qs) / 7

    def __call__(self, r):
        gaussians = jnp.exp(-((r[..., None] - self.mus) ** 2) / self.sigmas**2)
        return self.apply_envelope(r, gaussians)


class CombinedRadialBases(RadialBasis):
    r"""
    Class combining multiple radial bases.

    The total number of basis functions
    :math:`n_{rbf}` has to match the sum of the numbers of basis functions of
    the constituents.

    Args:
        n_rbf (int): the total number of basis functions
        r_cut (float): the cutoff radius for the radial bases
        n_rbfs (Sequence[int]): the number of basis function per basis respectively
        factories (Sequence[Callable]): the factories of the bases respectively
        kwargs: (Sequence[dict]): optional, kwargs for the bases respectively
    """

    def __init__(
        self,
        n_rbf: int,
        r_cut: float,
        n_rbfs: Sequence[int],
        factories: Sequence,
        kwargs: Optional[Sequence[dict]] = None,
    ):
        if not len(n_rbfs) == len(factories):
            raise ValueError(
                f'length of n_rbfs ({len(n_rbfs)}) should be equal to '
                f'length of factories ({len(factories)})'
            )
        if not sum(n_rbfs) == n_rbf:
            raise ValueError(
                f'n_rbf ({n_rbf}) should be equal to sum of n_rbfs ({sum(n_rbfs)})'
            )
        self.bases = []
        kwargs = kwargs or [{} for _ in factories]
        for n_rbf, factory, kwargs in zip(n_rbfs, factories, kwargs):
            self.bases.append(factory(n_rbf, r_cut, **(kwargs or {})))

    def __call__(self, r):
        basis = []
        for base in self.bases:
            basis.append(base(r))
        return jnp.concatenate(basis, axis=-1)


class EdgeFeatures:
    r"""
    Class combining the radial and angular bases to obtain edge features.

    Args:
        n_rbf (int): the number of radial basis functions
        r_cut (float): the cutoff radius for the radial basis
        irreps (Sequence[e3nn_jax.Irrep]): the irreducible representations for
            the angular basis
        separate (bool): default :data:`False`, whether to keep radial and angular
            parts separate.
        equivariant: default True, if False the distinction between feature
            dimension and irreps dimension is removed
        radial_basis_factory (Callable): optional, creates the radial basis
        radial_basis_kwargs (dict): optional, kwargs for the radial basis
    """

    def __init__(
        self,
        n_rbf: int,
        r_cut: float,
        irreps: Sequence[e3nn.Irrep],
        equivariant: bool = True,
        combine_idxs=None,
        *,
        separate: bool = False,
        radial_basis_factory: Optional = None,
        radial_basis_kwargs: Optional[dict] = None,
    ):
        assert combine_idxs is None or not equivariant
        if radial_basis_factory is None:
            radial_basis_factory = BesselBasis
        self.radial_basis = radial_basis_factory(
            n_rbf, r_cut, **(radial_basis_kwargs or {})
        )
        self.angular_basis = lambda r: e3nn.spherical_harmonics(
            irreps, r, normalize=True, normalization='component'
        ).array
        self.equivariant = equivariant
        self.combine_idxs = range(n_rbf) if combine_idxs is None else combine_idxs
        self.separate = separate

    def __call__(self, r: jnp.ndarray):
        radial = self.radial_basis(jnp.linalg.norm(r, axis=-1))
        angular = self.angular_basis(r)
        if self.equivariant:
            if self.separate:
                features = {'radial': radial, 'angular': angular}
            else:
                features = radial[..., None] * angular[..., None, :]
        else:
            features = [
                r[..., None] * (angular if i in self.combine_idxs else 1)
                for i, r in enumerate(jnp.rollaxis(radial, -1))
            ]
            features = tree_reduce(
                lambda x, y: jnp.concatenate((x, y), axis=-1),
                features,
            )
        return features


class PauliNetEdgeFeatures(EdgeFeatures):
    r"""Utility class to retain the old edge feature interface."""

    def __init__(
        self,
        feature_dim,
        cutoff=10.0,
        offset=True,
        powers=None,
        eps=1e-2,
        difference=False,
        envelope_factory=None,
    ):
        powers = [] if powers is None else powers
        irreps = [e3nn.Irrep('1y')]
        combine_idxs = []
        if difference:
            feature_dim -= 2
            powers = [*powers, 1]
            combine_idxs = [feature_dim - 1]
        envelope_factory = envelope_factory or IdentityEnvelope
        if feature_dim - len(powers) > 0:
            radial_basis_kwargs = {
                'n_rbfs': [feature_dim - len(powers)],
                'factories': [GaussianBasis],
                'kwargs': [{'offset': offset, 'envelope_factory': envelope_factory}],
            }
        if powers:
            radial_basis_kwargs['n_rbfs'] += [len(powers)]
            radial_basis_kwargs['factories'] += [DistancePowersBasis]
            radial_basis_kwargs['kwargs'] += [
                {'powers': powers, 'eps': eps, 'envelope_factory': envelope_factory}
            ]

        super().__init__(
            feature_dim,
            cutoff,
            irreps,
            False,
            combine_idxs,
            radial_basis_factory=CombinedRadialBases,
            radial_basis_kwargs=radial_basis_kwargs,
        )
