from functools import partial

import torch
from torch import nn

from deepqmc.torchext import SSP, get_log_dnn

from .schnet import ElectronicSchNet, SubnetFactory

__version__ = '0.3.0'
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

    def __init__(
        self, embedding_dim, activation_factory=SSP, *, n_layers=3, sum_first=True
    ):
        super().__init__()
        self.net = get_log_dnn(embedding_dim, 1, activation_factory, n_layers=n_layers)
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
        activation_factory=SSP,
        *,
        n_layers=3,
    ):
        super().__init__()
        nets = [
            get_log_dnn(
                embedding_dim,
                n_orbitals,
                activation_factory,
                n_layers=n_layers,
                last_bias=True,
            )
            for _ in range(n_backflows)
        ]
        self.nets = nn.ModuleList(nets)

    def forward(self, xs):
        return torch.stack([net(xs) for net in self.nets], dim=1)


class SchNetMeanFieldLayer(nn.Module):
    def __init__(self, factory, n_up):
        super().__init__()
        self.w = factory.w_subnet()
        self.g = factory.g_subnet()

    def forward(self, x, Y, edges_elec, edges_nuc):
        z_nuc = (self.w(edges_nuc) * Y[..., None, :, :]).sum(dim=-2)
        return self.g(z_nuc)


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
        mb_embedding_dim (int): dimension of many-body SchNet embeddings
        mf_embedding_dim (int): dimension of mean-field SchNet embeddings
        jastrow (str): type of Jastrow -- :data:`None`, ``'mean-field'``, or
            ``'many-body'``
        jastrow_kwargs (dict): extra arguments passed to :class:`Jastrow`
        backflow (str): type of backflow -- :data:`None`, ``'mean-field'``, or
            ``'many-body'``
        backflow_kwargs (dict): extra arguments passed to :class:`Backflow`
        schnet_kwargs (dict): extra arguments passed to :class:`ElectronicSchNet`
        subnet_kwargs (dict): extra arguments passed to :class:`SubnetFactory`
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
        *,
        mb_embedding_dim=128,
        mf_embedding_dim=128,
        jastrow='many-body',
        jastrow_kwargs=None,
        backflow='many-body',
        backflow_kwargs=None,
        schnet_kwargs=None,
        subnet_kwargs=None,
        mf_schnet_kwargs=None,
        mf_subnet_kwargs=None,
    ):
        assert not jastrow or jastrow in ['mean-field', 'many-body']
        assert not backflow or backflow in ['mean-field', 'many-body']
        super().__init__()
        self.schnet = (
            ElectronicSchNet(
                n_up,
                n_down,
                n_atoms,
                embedding_dim=mb_embedding_dim,
                subnet_metafactory=partial(SubnetFactory, **(subnet_kwargs or {})),
                **(schnet_kwargs or {}),
            )
            if 'many-body' in [jastrow, backflow]
            else None
        )
        self.mf_schnet = (
            MeanFieldElectronicSchNet(
                n_up,
                n_down,
                n_atoms,
                embedding_dim=mf_embedding_dim,
                subnet_metafactory=partial(SubnetFactory, **(mf_subnet_kwargs or {})),
                **(mf_schnet_kwargs or {}),
            )
            if 'mean-field' in [jastrow, backflow]
            else None
        )
        embedding_dim = {'mean-field': mf_embedding_dim, 'many-body': mb_embedding_dim}
        self.jastrow_type = jastrow
        if jastrow:
            self.jastrow = Jastrow(embedding_dim[jastrow], **(jastrow_kwargs or {}))
        self.backflow_type = backflow
        if backflow:
            self.backflow = Backflow(
                embedding_dim[backflow],
                n_orbitals,
                n_backflows,
                **(backflow_kwargs or {}),
            )

    def forward(self, dists_nuc, dists_elec):
        embeddings = {}
        if self.mf_schnet:
            embeddings['mean-field'] = self.mf_schnet(dists_nuc)
        if self.schnet:
            embeddings['many-body'] = self.schnet(dists_elec, dists_nuc)
        jastrow = (
            self.jastrow(embeddings[self.jastrow_type]) if self.jastrow_type else None
        )
        backflow = (
            self.backflow(embeddings[self.backflow_type])
            if self.backflow_type
            else None
        )
        return jastrow, backflow
