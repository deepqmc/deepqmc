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
        n_orb_up (int): the number of spin-up orbitals in a single deterimant,
            to compute backflow factors for. This is equal to the number of
            spin-up electrons, except when full determinants are used, in which case
            it is equal to the total number of electrons.
        n_orb_down (int): the number of spin-up orbitals in a single deterimant,
            to compute backflow factors for. This is equal to the number of
            spin-up electrons, except when full determinants are used, in which case
            it is equal to the total number of electrons.
        n_determinants (int): the number of determinants to use.
        n_backflows (int): the number of independent backflow channels for each orbital,
            e.g. two channels are necessary if both additive and multiplicative
            backflows are used.
        gnn_factory (Callable): function that returns a GNN instance.
        jastrow_factory (Callable): function that returns a :class:`Jastrow` instance.
        backflow_factory (Callable): function that returns a :class:`Backflow` instance.
        embedding_dim (int): the length of the electron embedding vectors.
    """

    def __init__(
        self,
        mol,
        n_orb_up,
        n_orb_down,
        n_determinants,
        n_backflows,
        *,
        gnn_factory=None,
        jastrow_factory=None,
        backflow_factory=None,
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
            self.backflow = {
                l: backflow_factory(embedding_dim, n_determinants * n, n_backflows)
                for l, n in zip(['up', 'down'], [n_orb_up, n_orb_down])
            }
        else:
            self.backflow = None

    def __call__(self, phys_conf):
        if self.gnn:
            embeddings = self.gnn(phys_conf)
        jastrow = self.jastrow(embeddings) if self.jastrow else None
        backflow = (
            (
                self.backflow['up'](embeddings[..., : self.n_up, :]),
                self.backflow['down'](embeddings[..., self.n_up :, :]),
            )
            if self.backflow
            else None
        )
        return jastrow, backflow
