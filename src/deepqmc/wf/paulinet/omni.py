from functools import partial

import numpy as np
import torch
from torch import nn

from deepqmc.physics import (
    pairwise_diffs,
    pairwise_distance,
    pairwise_self_distance,
    pairwise_self_difference,
)
from deepqmc.torchext import SSP, get_mlp, idx_perm

from .schnet import ElectronicSchNet, SubnetFactory

__version__ = '0.5.0'
__all__ = ['OmniSchNet']


class Jastrow(nn.Module):
    r"""Jastrow network :math:`\eta_{\boldsymbol \theta}`.

    The Jastrow factor consists of a vanilla neural network with logarithmically
    progressing layer widths that maps the electronic embeddings to the final
    Jastrow factor,

    .. math::
        J :=
        \begin{cases}
        \eta_{\boldsymbol \theta}\big(\textstyle\sum_i \mathbf x_i^
        {(L)}\big) & \text{if }\texttt{sum\_first}\\
        \textstyle\sum_i\eta_{\boldsymbol \theta}\big(\mathbf x_i^
        {(L)}\big) & \text{otherwise}
        \end{cases}

    Args:
        embedding_dim (int):  :math:`\dim(\mathbf x_i^{(L)})`,
            dimension of electronic embedding input
        activation_factory (callable): creates activation functions between
            layers
        n_layers (int): number of neural network layers
        sum_first (bool): whether embeddings are summed before passed to the
            network

    Shape:
        - Input, :math:`\mathbf x_i^{(L)}`: :math:`(*,N,D)`
        - Output, :math:`J`: :math:`(*)`

    Attributes:
        net: :class:`torch.nn.Sequential` representing vanilla neural network
    """

    def __init__(self, embedding_dim, *, n_layers=3, sum_first=True, **kwargs):
        kwargs.setdefault('activation', SSP)
        kwargs.setdefault('hidden_layers', ('log', n_layers))
        super().__init__()
        self.net = get_mlp(embedding_dim, 1, **kwargs)
        self.sum_first = sum_first

    def forward(self, xs):
        if self.sum_first:
            xs = self.net(xs.sum(dim=-2))
        else:
            xs = self.net(xs).sum(dim=-2)
        return xs.squeeze(dim=-1)


class Backflow(nn.Module):
    r"""Represents backflow networks :math:`\boldsymbol\kappa_{\boldsymbol\theta,q}`.

    The backflow transformation consists of :math:`N_\text{bf}` vanilla neural
    networks with logarithmically progressing layer width maping the electronic
    embeddings to the backflow transformations,

    .. math::
        \mathbf f_i := \mathbf \kappa_{\boldsymbol \theta}\big(\mathbf x_i^{(L,
        \text{mf/bf})}\big)

    Args:
        embedding_dim (int): :math:`\dim(\mathbf x_i^{(L)})`,
            dimension of electronic embedding input
        n_orbitals (int): :math:`N_\text{orb}` number of orbitals
        n_backflows (int): :math:`N_\text{bf}` number of backflows
        activation_factory (callable): creates activation functions between
            layers
        n_layers (int): number of neural network layers

    Shape:
        - Input, :math:`\mathbf x_i^{(L)}`: :math:`(*,N,D)`
        - Output, :math:`f_{q\mu i}`: :math:`(*,N_\text{bf},N,N_\text{orb})`

    Attributes:
        nets: :class:`torch.nn.ModuleList` containing :math:`N_text{bf}`
            vanilla neural networks
    """

    def __init__(
        self,
        embedding_dim,
        n_orbitals,
        n_backflows,
        multi_head=True,
        *,
        n_layers=3,
        **kwargs,
    ):
        kwargs.setdefault('activation', SSP)
        kwargs.setdefault('hidden_layers', ('log', n_layers))
        kwargs.setdefault('last_bias', True)
        super().__init__()
        self.multi_head = multi_head
        if multi_head:
            mlps = [
                get_mlp(embedding_dim, n_orbitals, **kwargs) for _ in range(n_backflows)
            ]
            self.mlps = nn.ModuleList(mlps)
        else:
            self.n_orbitals = n_orbitals
            hidden_layers = kwargs.pop('hidden_layers')
            self.mlp = nn.Sequential(
                get_mlp(embedding_dim, hidden_layers[-1], hidden_layers[:-1], **kwargs),
                nn.Linear(hidden_layers[-1], n_backflows * n_orbitals),
            )

    def forward(self, xs):
        if self.multi_head:
            return torch.stack([net(xs) for net in self.mlps], dim=1)
        else:
            xs = self.mlp(xs)
            xs = xs.unflatten(-1, (-1, self.n_orbitals))
            xs = xs.transpose(-2, -3)
            return xs


class RealSpaceBackflow(nn.Module):
    def __init__(self, embedding_dim, nuc_charges, decay_type, **kwargs):
        super().__init__()
        self.decay_type = decay_type
        if nuc_charges is not None:
            self.register_buffer('nuc_charges', torch.tensor(nuc_charges))
        self.mlp = nn.ModuleDict(
            {l: get_mlp(embedding_dim, 1, **kwargs) for l in ['el', 'nuc']}
        )

    def forward(self, rs, coords, messages, embeddings):
        n_elec, n_nuc = rs.shape[-2], len(coords)
        messages_el, messages_nuc = messages
        embeddings = embeddings['many-body'][:, :, None]
        f_el = torch.cat(
            [embeddings.expand(-1, -1, n_elec - 1, -1), messages_el], dim=-1
        )
        f_nuc = torch.cat([embeddings.expand(-1, -1, n_nuc, -1), messages_nuc], dim=-1)
        i, j = idx_perm(n_elec, 2, rs.device)
        diffs_nuc = pairwise_diffs(rs, coords)
        diffs = torch.cat([pairwise_diffs(-rs, -rs)[:, i, j], diffs_nuc], dim=-2)
        f = torch.cat([self.mlp['el'](f_el), self.mlp['nuc'](f_nuc)], dim=-2)
        ps = f * diffs[..., :3] / (1 + diffs[..., 3:] ** (3 / 2))
        ps = ps.sum(dim=-2)
        if self.decay_type == 'rios':
            R = diffs_nuc[..., -1].sqrt().min(dim=-1).values / 0.5
            decay = torch.where(
                R < 1, R ** 2 * (6 - 8 * R + 3 * R ** 2), R.new_tensor(1)
            )
        elif self.decay_type == 'deeperwin':
            decay = (
                (diffs_nuc[..., -1] / (0.5 / self.nuc_charges) ** 2).tanh().prod(dim=-1)
            )
        ps = np.exp(-3.5) * decay[..., None] * ps
        return ps


class SchNetMeanFieldLayer(nn.Module):
    def __init__(self, factory, n_up):
        super().__init__()
        self.w = factory.w_subnet()
        self.g = factory.g_subnet()

    def forward(self, x, Y, edges_elec, edges_nuc):
        z_nuc = (self.w(edges_nuc) * Y[..., None, :, :]).sum(dim=-2)
        return self.g(z_nuc), None


class MeanFieldElectronicSchNet(ElectronicSchNet):
    r"""Mean-field variant of :class:`ElectronicSchNet`.

    This mean-field variant of the graph neural nework :class:`ElectronicSchNet`
    uses :class:`SchNetMeanFieldLayer` as default, removing electronic
    interactions and returning mean-field electronic embeddings. In contrast
    to :class:`ElectronicSchNet` the :meth:`forward` only uses nuclear edges.

    """

    LAYER_FACTORIES = {'mean-field': SchNetMeanFieldLayer}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, version='mean-field', **kwargs)

    def forward(self, dists_nuc):
        *batch_dims, n_elec = dists_nuc.shape[:-1]
        dists_elec_dummy = dists_nuc.new_empty(*batch_dims, n_elec, n_elec)
        return super().forward(dists_elec_dummy, dists_nuc)


class OmniSchNet(nn.Module):
    r"""Combined Jastrow/backflow neural network based on SchNet.

    This module uses an instance of :class:`ElectronicSchNet` to build a
    many-body or a mean-field feature representation of electrons, which are
    subsequently passed as an input into additional trainable functions to
    obtain many-body or mean-field Jastrow factor and backflow transformations.
    The mean-field embeddings are obtained with a variant of
    :class:`ElectronicSchNet` with the electron--electron message passing omitted.
    The type of embeddings used for Jastrow and backflow can be chosen
    individually and equivalent embbedings are shared. The module is used to
    generate the Jastrow factor and backflow transformation within
    :class:`~deepqmc.wf.PauliNet`.

    The Jastrow factor and backflow are obtained as

    .. math::
        J:=\eta_{\boldsymbol\theta}\big(\textstyle\sum_i\mathbf
        x_i^{(L)}\big),\qquad
        f_{q\mu i}(\mathbf r)
        :=\Big(\boldsymbol\kappa_{\boldsymbol\theta,q}\big(\mathbf
        x_i^{(L)}\big)\Big)_\mu

    where :math:`\eta_{\boldsymbol\theta}` and
    :math:`\boldsymbol\kappa_{\boldsymbol\theta,q}` are vanilla deep
    neural networks and :math:`\mathbf x_i^{(L)}` are either the many-body or
    mean-field embedding.

    Args:
        n_atoms (int): :math:`M`, number of atoms
        n_up (int): :math:`N^\uparrow`, number of spin-up electrons
        n_down (int): :math:`N^\downarrow`, number of spin-down electrons
        n_orbitals (int): :math:`N_\text{orb}`, number of molecular orbitals
        n_backflows (int): :math:`N_\text{bf}`, number of backflow channnels
        embedding_dim (int): dimension of many-body SchNet embeddings
        jastrow (str): type of Jastrow -- :data:`None`, ``'mean-field'``, or
            ``'many-body'``
        jastrow_kwargs (dict): extra arguments passed to :class:`Jastrow`
        backflow (str): type of backflow -- :data:`None`, ``'mean-field'``, or
            ``'many-body'``
        backflow_kwargs (dict): extra arguments passed to :class:`Backflow`
        schnet_kwargs (dict): extra arguments passed to :class:`ElectronicSchNet`
        subnet_kwargs (dict): extra arguments passed to :class:`SubnetFactory`
        mf_embedding_dim (int): dimension of mean-field SchNet embeddings
        mf_schnet_kwargs (dict): extra arguments passed to the mean-field variant
            of :class:`ElectronicSchNet`
        mf_subnet_kwargs (dict): extra arguments passed to :class:`SubnetFactory`

    Shape:
        - Input1, :math:`\lvert\mathbf r_i-\mathbf r_j\rvert`: :math:`(*,N,N)`
        - Input2, :math:`\lvert\mathbf r_i-\mathbf R_I\rvert`: :math:`(*,N,M)`
        - Output1, :math:`J`: :math:`(*)`
        - Output2, :math:`f_{q\mu i}`: :math:`(*,N_\text{bf},N,N_\text{orb})`

    Attributes:
        schnet: :class:`ElectronicSchNet` network
        mf_schnet: mean-field variant of :class:`ElectronicSchNet` network
        jastrow: :class:`Jastrow` network
        backflow: :class:`Backflow` network
    """

    def __init__(
        self,
        n_atoms,
        n_up,
        n_down,
        n_orbitals,
        n_backflows,
        schnet_factory=None,
        jastrow_factory=None,
        backflow_factory=None,
        rs_backflow_factory=None,
        *,
        embedding_dim=128,
        jastrow='many-body',
        jastrow_kwargs=None,
        backflow='many-body',
        backflow_kwargs=None,
        rs_backflow=None,
        rs_backflow_kwargs=None,
        schnet_kwargs=None,
        subnet_kwargs=None,
        mf_embedding_dim=128,
        mf_schnet_kwargs=None,
        mf_subnet_kwargs=None,
        spin_jastrow=False,
    ):
        assert not jastrow or jastrow in ['mean-field', 'many-body']
        assert not backflow or backflow in ['mean-field', 'many-body']
        assert not rs_backflow or rs_backflow in ['many-body']
        super().__init__()
        self.n_up = n_up
        if not schnet_factory:
            schnet_factory = partial(
                ElectronicSchNet,
                subnet_metafactory=partial(SubnetFactory, **(subnet_kwargs or {})),
                **(schnet_kwargs or {}),
            )
        self.schnet = (
            schnet_factory(n_up, n_down, n_atoms, embedding_dim)
            if 'many-body' in [jastrow, backflow]
            else None
        )
        self.mf_schnet = (
            MeanFieldElectronicSchNet(
                n_up,
                n_down,
                n_atoms,
                mf_embedding_dim,
                subnet_metafactory=partial(SubnetFactory, **(mf_subnet_kwargs or {})),
                **(mf_schnet_kwargs or {}),
            )
            if 'mean-field' in [jastrow, backflow]
            else None
        )
        embedding_dim = {'mean-field': mf_embedding_dim, 'many-body': embedding_dim}
        self.jastrow_type = jastrow
        if jastrow:
            if not jastrow_factory:
                jastrow_factory = partial(Jastrow, **(jastrow_kwargs or {}))
            self.jastrow = (
                nn.ModuleDict(
                    {l: jastrow_factory(embedding_dim[jastrow]) for l in ['up', 'down']}
                )
                if spin_jastrow
                else jastrow_factory(embedding_dim[jastrow])
            )
        self.backflow_type = backflow
        if backflow:
            if not backflow_factory:
                backflow_factory = partial(Backflow, **(backflow_kwargs or {}))
            self.backflow = (
                backflow_factory(embedding_dim[backflow], n_orbitals, n_backflows)
                if isinstance(n_orbitals, int)
                else nn.ModuleDict(
                    {
                        l: backflow_factory(embedding_dim[backflow], n, n_backflows)
                        for l, n in zip(['up', 'down'], n_orbitals)
                    }
                )
            )
        self.rs_backflow_type = rs_backflow
        if rs_backflow:
            if not rs_backflow_factory:
                rs_backflow_factory = partial(
                    RealSpaceBackflow, **(rs_backflow_kwargs or {})
                )
            self.rs_backflow = rs_backflow_factory(
                embedding_dim[rs_backflow] + self.schnet.kernel_dim
            )

    def forward(self, rs, coords):
        dists_elec = pairwise_self_distance(rs, full=True)
        dists_nuc = pairwise_distance(rs, coords)
        diffs_elec = pairwise_self_difference(rs, full=True)
        diffs_nuc = pairwise_diffs(rs, coords)[..., :3]
        embeddings = {}
        if self.mf_schnet:
            embeddings['mean-field'], _ = self.mf_schnet(dists_nuc)
        if self.schnet:
            embeddings['many-body'], messages = self.schnet(
                dists_elec, dists_nuc, diffs_elec, diffs_nuc
            )
        if self.jastrow_type:
            if isinstance(self.jastrow, nn.ModuleDict):
                j_up = self.jastrow['up'](
                    embeddings[self.jastrow_type][..., : self.n_up, :]
                )
                j_down = self.jastrow['down'](
                    embeddings[self.jastrow_type][..., self.n_up :, :]
                )
                jastrow = (j_up + j_down) / 2
            else:
                jastrow = self.jastrow(embeddings[self.jastrow_type])
        else:
            jastrow = None
        backflow = (
            None
            if not self.backflow_type
            else (
                self.backflow['up'](
                    embeddings[self.backflow_type][..., : self.n_up, :]
                ),
                self.backflow['down'](
                    embeddings[self.backflow_type][..., self.n_up :, :]
                ),
            )
            if isinstance(self.backflow, nn.ModuleDict)
            else self.backflow(embeddings[self.backflow_type])
        )
        ps = (
            self.rs_backflow(rs, coords, messages, embeddings)
            if self.rs_backflow_type
            else None
        )
        return jastrow, backflow, ps
