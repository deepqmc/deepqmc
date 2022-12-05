from functools import partial

import haiku as hk
import jax.numpy as jnp

from ...gnn import SchNet
from ...hkext import MLP, ssp
from ...utils import unflatten


class Jastrow(hk.Module):
    r"""The deep Jastrow factor.

    Args:
        embedding_dim (int): the length of the electron embedding vectors.
        n_layers (int): the number of Jastrow MLP layers.
        sum_first (bool): if :data:`True`, the electronic embeddings are summed before
            feeding them to the MLP. Otherwise the MLP is applyied separately on each
            electron embedding, and the outputs are summed, yielding a (quasi)
            mean-field Jastrow factor.
        name (str): the name of this haiku module.
    """

    def __init__(
        self, embedding_dim, *, n_layers=3, sum_first=True, name='Jastrow', **kwargs
    ):
        kwargs.setdefault('activation', ssp)
        kwargs.setdefault('hidden_layers', ('log', n_layers))
        kwargs.setdefault('last_linear', True)
        kwargs.setdefault('bias', 'not_last')
        super().__init__(name=name)
        self.net = MLP(embedding_dim, 1, **kwargs)
        self.sum_first = sum_first

    def __call__(self, xs):
        if self.sum_first:
            xs = self.net(xs.sum(axis=-2))
        else:
            xs = self.net(xs).sum(axis=-2)
        return xs.squeeze(axis=-1)


class Backflow(hk.Module):
    r"""The deep backflow factor.

    Args:
        embedding_dim (int): the length of the electron embedding vectors.
        n_orbitals (int): the number of orbitals to compute backflow factors for.
        n_backflow (int): the number of independent backflow factors for each orbital.
        multi_head (bool): if :data:`True`, create separate MLPs for the
            :data:`n_backflow` many backflows, otherwise use a single larger MLP
            for all.
        n_layers (int): the number of layers in the MLP(s).
        name (str): the name of this haiku module.
        param_scaling (float): a scaling factor to apply during the initialization of
            the MLP parameters.
    """

    def __init__(
        self,
        embedding_dim,
        n_orbitals,
        n_backflows=1,
        multi_head=True,
        *,
        n_layers=3,
        name='Backflow',
        param_scaling=1.0,
        **kwargs,
    ):
        kwargs.setdefault('activation', ssp)
        kwargs.setdefault('hidden_layers', ('log', n_layers))
        kwargs.setdefault('last_linear', True)
        super().__init__(name=name)
        self.multi_head = multi_head
        if multi_head:
            self.nets = [
                MLP(
                    embedding_dim,
                    n_orbitals,
                    w_init=hk.initializers.VarianceScaling(param_scaling),
                    **kwargs,
                )
                for _ in range(n_backflows)
            ]
        else:
            self.n_orbitals = n_orbitals
            hidden_layers = kwargs.pop('hidden_layers')
            self.net = MLP(
                embedding_dim,
                n_backflows * n_orbitals,
                hidden_layers,
                w_init=hk.initializers.VarianceScaling(param_scaling),
                **kwargs,
            )

    def __call__(self, xs):
        if self.multi_head:
            return jnp.stack([net(xs) for net in self.nets], axis=-3)
        else:
            xs = self.net(xs)
            xs = unflatten(xs, -1, (-1, self.n_orbitals))
            xs = xs.swapaxes(-2, -3)
            return xs


class OmniNet(hk.Module):
    r"""Combine the GNN, the Jastrow and backflow MLPs.

    A GNN is used to create embedding vectors for each electron, which are then fed
    into the Jastrow and/or backflow MLPs to produce the Jastrow--backflow part of
    deep QMC Ansatzes.

    Args:
        mol (~deepqmc.Molecule): the molecule to consider.
        n_orbitals (int): the number of orbitals to compute backflow factors for.
        n_backflow (int): the number of independent backflow factors for each orbital.
        gnn_factory (Callable): optional, a function that returns a GNN instance.
        jastrow_factory (Callable): optional a function that returns a :class:`Jastrow`
            instance.
        backflow_factory (Callable): optional a function that returns a
            :class:`Backflow` instance.
        embedding_dim (int): the length of the electron embedding vectors.
        gnn_kwargs (dict): optional, additional keyword arguments passed to
            :data:`gnn_factory`.
        jastrow (bool): whether a Jastrow factor is computed.
        jastrow_kwargs (dict): optional, additional keyword arguments passed to
            :data:`jastrow_factory`.
        backflow (bool): whether backflow factors are computed.
        backflow_kwargs (dict): optional, additional keyword arguments passed to
            :data:`backflow_factory`.
    """

    def __init__(
        self,
        mol,
        n_orbitals,
        n_backflows,
        gnn_factory=None,
        jastrow_factory=None,
        backflow_factory=None,
        *,
        embedding_dim=128,
        gnn_kwargs=None,
        jastrow=True,
        jastrow_kwargs=None,
        backflow=True,
        backflow_kwargs=None,
    ):
        super().__init__()
        self.n_up = mol.n_up
        if jastrow or backflow:
            if gnn_factory is None:
                gnn_factory = SchNet
            self.gnn = gnn_factory(
                mol,
                embedding_dim,
                **(gnn_kwargs or {}),
            )
        else:
            self.gnn = None

        if jastrow:
            if jastrow_factory is None:
                jastrow_factory = partial(Jastrow, **(jastrow_kwargs or {}))
            self.jastrow = jastrow_factory(embedding_dim)
        else:
            self.jastrow = None

        if backflow:
            if backflow_factory is None:
                backflow_factory = partial(Backflow, **(backflow_kwargs or {}))
            self.backflow = (
                backflow_factory(embedding_dim, n_orbitals, n_backflows)
                if not isinstance(n_orbitals, tuple)
                else {
                    l: backflow_factory(
                        embedding_dim, n, n_backflows, name=f'Backflow_{l}'
                    )
                    for l, n in zip(['up', 'down'], n_orbitals)
                }
            )
        else:
            self.backflow = None

    def __call__(self, rs):
        if self.gnn:
            embeddings = self.gnn(rs)
        jastrow = self.jastrow(embeddings) if self.jastrow else None
        backflow = (
            None
            if not self.backflow
            else (
                self.backflow['up'](embeddings[..., : self.n_up, :]),
                self.backflow['down'](embeddings[..., self.n_up :, :]),
            )
            if isinstance(self.backflow, dict)
            else self.backflow(embeddings)
        )
        return jastrow, backflow
