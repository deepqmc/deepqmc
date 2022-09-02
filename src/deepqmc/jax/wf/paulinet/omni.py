from functools import partial

import haiku as hk
import jax.numpy as jnp

from ...hkext import MLP, SSP
from ...jaxext import unflatten
from .schnet import SchNet


class Jastrow(hk.Module):
    def __init__(
        self, embedding_dim, *, n_layers=3, sum_first=True, name='Jastrow', **kwargs
    ):
        kwargs.setdefault('activation', SSP)
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
    def __init__(
        self,
        embedding_dim,
        n_orbitals,
        n_backflows=1,
        multi_head=True,
        *,
        n_layers=3,
        name='Backflow',
        **kwargs,
    ):
        kwargs.setdefault('activation', SSP)
        kwargs.setdefault('hidden_layers', ('log', n_layers))
        kwargs.setdefault('last_linear', True)
        super().__init__(name=name)
        self.multi_head = multi_head
        if multi_head:
            self.nets = [
                MLP(embedding_dim, n_orbitals, **kwargs) for _ in range(n_backflows)
            ]
        else:
            self.n_orbitals = n_orbitals
            hidden_layers = kwargs.pop('hidden_layers')
            self.net = MLP(
                embedding_dim, n_backflows * n_orbitals, hidden_layers, **kwargs
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
    def __init__(
        self,
        mol,
        n_orbitals,
        n_backflows,
        gnn_factory=None,
        jastrow_factory=None,
        backflow_factory=None,
        rs_backflow_factory=None,
        *,
        embedding_dim=128,
        occupancy=10,
        gnn_kwargs=None,
        jastrow=True,
        jastrow_kwargs=None,
        backflow=True,
        backflow_kwargs=None,
        rs_backflow=False,
        rs_backflow_kwargs=None,
        subnet_kwargs=None,
    ):
        super().__init__()
        self.n_up = mol.n_up
        if jastrow or backflow or rs_backflow:
            if gnn_factory is None:
                gnn_factory = SchNet
            self.gnn = gnn_factory(
                len(mol.coords),
                mol.n_up,
                mol.n_down,
                mol.coords,
                embedding_dim,
                **(gnn_kwargs or {}),
            )
        else:
            self.gnn = None

        if jastrow_factory is None:
            jastrow_factory = partial(Jastrow, **(jastrow_kwargs or {}))
        self.jastrow = jastrow_factory(embedding_dim) if jastrow else None

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
