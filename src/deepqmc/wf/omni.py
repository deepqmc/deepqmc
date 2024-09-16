import math
from collections.abc import Callable

import haiku as hk
import jax
import jax.numpy as jnp

from ..hkext import GLU
from ..types import KeyArray
from ..utils import unflatten


class Jastrow(hk.Module):
    r"""The deep Jastrow factor.

    Args:
        sum_first (bool): if :data:`True`, the electronic embeddings are summed before
            feeding them to the MLP. Otherwise the MLP is applied separately on each
            electron embedding, and the outputs are summed, yielding a (quasi)
            mean-field Jastrow factor.
        name (str): the name of this haiku module.
    """

    def __init__(
        self,
        *,
        sum_first,
        subnet_factory: Callable[[int], Callable],
        name='Jastrow',
    ):
        super().__init__(name=name)
        self.net = subnet_factory(1)
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
        n_orbitals (int): the number of orbitals to compute backflow factors for.
        n_determinants (int): the number of determinants of the ansatz.
        n_backflow (int): the number of independent backflow factors for each orbital.
        multi_head (bool): if :data:`True`, create separate MLPs for the
            :data:`n_backflow` many backflows, otherwise use a single larger MLP
            for all.
        name (str): the name of this haiku module.
    """

    def __init__(
        self,
        n_orbitals,
        n_determinants,
        n_backflows,
        spin,
        multi_head=True,
        *,
        subnet_factory: Callable[[int], Callable],
        name='Backflow',
    ):
        super().__init__(name=name)
        self.multi_head = multi_head
        self.n_orbitals = n_orbitals
        self.n_determinants = n_determinants
        self.spin = spin
        if multi_head:
            self.nets = [
                subnet_factory(n_orbitals * n_determinants) for _ in range(n_backflows)
            ]
        else:
            self.net = subnet_factory(n_backflows * n_orbitals * n_determinants)

    def __call__(self, xs):
        if self.multi_head:
            xs = jnp.stack([net(xs) for net in self.nets], axis=-3)
        else:
            xs = self.net(xs)
            xs = unflatten(xs, -1, (-1, self.n_orbitals * self.n_determinants))
            xs = xs.swapaxes(-2, -3)
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
        n_orb_down (int): the number of spin-down orbitals in a single deterimant,
            to compute backflow factors for. This is equal to the number of
            spin-down electrons, except when full determinants are used, in which case
            it is equal to the total number of electrons.
        n_determinants (int): the number of determinants to use.
        n_backflows (int): the number of independent backflow channels for each orbital,
            e.g. two channels are necessary if both additive and multiplicative
            backflows are used.
        embedding_dim (int): the length of the electron embedding vectors.
        gnn_factory (~collections.abc.Callable): function that returns a GNN instance.
        jastrow_factory (~collections.abc.Callable): function that returns a
            :class:`Jastrow` instance.
        backflow_factory (~collections.abc.Callable): function that returns a
            :class:`Backflow` instance.
    """

    def __init__(
        self,
        hamil,
        n_orb_up,
        n_orb_down,
        n_determinants,
        n_backflows,
        *,
        embedding_dim,
        gnn_factory,
        jastrow_factory,
        backflow_factory,
        nuclear_gnn_head=None,
    ):
        super().__init__()
        self.n_up = hamil.n_up
        self.gnn = gnn_factory(hamil, embedding_dim) if gnn_factory else None
        self.jastrow = jastrow_factory() if jastrow_factory else None
        self.backflow = (
            {
                l: backflow_factory(n_orb, n_determinants, n_backflows, l)
                for l, n_orb in zip(['up', 'down'], [n_orb_up, n_orb_down])
            }
            if backflow_factory
            else None
        )
        self.nuclear_gnn_head = nuclear_gnn_head() if nuclear_gnn_head else None

    def __call__(self, phys_conf):
        if self.gnn:
            graph_nodes = self.gnn(phys_conf)
            embeddings = graph_nodes.electrons
            nucleus_embeddings = graph_nodes.nuclei
        else:
            return None, None, None
        nuclei_dependent_params = (
            self.nuclear_gnn_head(nucleus_embeddings) if self.nuclear_gnn_head else None
        )
        jastrow = self.jastrow(embeddings) if self.jastrow else None
        backflow = (
            (
                self.backflow['up'](embeddings[: self.n_up]),
                self.backflow['down'](embeddings[self.n_up :]),
            )
            if self.backflow
            else None
        )
        return jastrow, backflow, nuclei_dependent_params


class NuclearGNNHead(hk.Module):
    r"""A GNN head that predicts parameters from nucleus embeddings."""

    def __init__(self, *, one_particle_parameters):
        super().__init__()
        self.one_particle_readouts = {
            f'{k}_{spin}': self.one_particle_readout_factory(k, spin, per_nucleus_shape)
            for k, per_nucleus_shape in one_particle_parameters.items()
            for spin in ['up', 'down']
        }

    def one_particle_readout_factory(
        self, key: KeyArray, spin: str, per_nucleus_shape: tuple[int]
    ) -> Callable[[jax.Array], jax.Array]:
        def readout(embedding):
            glu_out = GLU(math.prod(per_nucleus_shape), name=f'{key}_readout_glu')(
                embedding, embedding
            ).reshape(-1, *per_nucleus_shape)
            return glu_out + hk.get_parameter(
                f'{key}_bias_{spin}',
                glu_out.shape,
                init=lambda s, d: 2 * jnp.ones(s, d),
            )

        return readout

    def __call__(self, nucleus_embeddings):
        return {
            k: readout(nucleus_embeddings)
            for k, readout in self.one_particle_readouts.items()
        }
