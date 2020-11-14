from functools import partial

import torch
from torch import nn

from deepqmc.torchext import SSP, get_log_dnn

from .distbasis import DistanceBasis
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

    def forward(self, edges_nuc):
        *batch_dims, n_elec = edges_nuc.shape[:-2]
        edges_elec_dummy = edges_nuc.new_empty(*batch_dims, n_elec, n_elec, 0)
        return super().forward(edges_elec_dummy, edges_nuc)


class OmniSchNet(nn.Module):
    r"""Combined Jastrow/backflow neural network based on SchNet.

    This module uses a single :class:`ElectronicSchNet` instance to build
    many-body feature representations of electrons, which are subsequently
    passed as an input into additional trainable functions to obtain the Jastrow
    factor and backflow transformations. This enables the use of a single SchNet
    module within :class:`~deepqmc.wf.PauliNet`, which itself makes no
    assumptions about parameter sharing by the Jastrow factor and backflow
    network modules.

    The Jastrow factor and backflow are obtained as

    .. math::
        J:=\eta_{\boldsymbol\theta}\big(\textstyle\sum_i\mathbf x_i^{(L)}\big),\qquad
        \kappa_{q\mu i}(\mathbf r)
        :=\Big(\boldsymbol\kappa_{\boldsymbol\theta,q}\big(\mathbf
        x_i^{(L)}\big)\Big)_\mu

    where :math:`\eta_{\boldsymbol\theta}` and
    :math:`\boldsymbol\kappa_{\boldsymbol\theta,q}` are vanilla deep neural networks.

    The Jastrow and backflow are obtained by calling :meth:`forward_jastrow` and
    :meth:`forward_backflow`, which calls SchNet only once under the hood. After
    the forward pass is finished, :math:`close` must be called, which ensures
    that SchNet is called anew in the next pass.

    Args:
        mol (:class:`~deepqmc.Molecule`): molecule whose wave function is represented
        dist_feat_dim (int): passed to :class:`ElectronicSchNet`
        n_up (int): passed to :class:`ElectronicSchNet`
        n_down (int): passed to :class:`ElectronicSchNet`
        n_orbitals (int): :math:`N_\text{orb}`, number of molecular orbitals
        n_channels (int): :math:`C`, number of backflow channels
        embedding_dim (int): :math:`\dim(\mathbf x_i^{(L)})`, dimension of SchNet
            embeddings
        with_jastrow (bool): if false, the Jastrow part is void
        n_jastrow_layers (int): number of layers in the Jastrow factor network
        with_backflow (bool): if false, the backflow part is void
            :math:`\tilde\varphi_{\mu i}(\mathbf r):=\varphi_\mu(\mathbf r_i)`
        n_backflow_layers (int): number of layers in the backflow network
        schnet_kwargs (dict): extra arguments passed to :class:`ElectronicSchNet`
        subnet_kwargs (dict): extra arguments passed to :class:`SubnetFactory`
        mf_schnet_kwargs (dict): extra arguments passed to
            :class:`MeanFieldElectronicSchNet`
        mf_subnet_kwargs (dict): extra arguments passed to :class:`SubnetFactory`

    Shape:
        - :meth:`forward_jastrow`:
            - Input1, :math:`\mathbf e(\lvert\mathbf r_i-\mathbf r_j\rvert)`:
              :math:`(*,N,N,\dim(\mathbf e))`
            - Input2, :math:`\mathbf e(\lvert\mathbf r_i-\mathbf R_I\rvert)`:
              :math:`(*,N,M,\dim(\mathbf e))`
            - Output, :math:`J`: :math:`(*)`
        - :meth:`forward_backflow`:
            - Input1, :math:`\mathbf e(\lvert\mathbf r_i-\mathbf r_j\rvert)`:
              :math:`(*,N,N,\dim(\mathbf e))`
            - Input2, :math:`\mathbf e(\lvert\mathbf r_i-\mathbf R_I\rvert)`:
              :math:`(*,N,M,\dim(\mathbf e))`
            - Output, :math:`\kappa_{q\mu i}`:
              :math:`(*,C,N,N_\text{orb})`

    Attributes:
        schnet: :class:`ElectronicSchNet` network
        jastrow: :class:`torch.nn.Sequential` representing a vanilla DNN
        backflow: :class:`torch.nn.Sequential` representing a vanilla DNN
    """

    def __init__(
        self,
        n_atoms,
        n_up,
        n_down,
        n_orbitals,
        n_backflows,
        *,
        dist_feat_dim=32,
        dist_feat_cutoff=10.0,
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
        assert jastrow in [None, 'mean-field', 'many-body']
        assert backflow in [None, 'mean-field', 'many-body']
        super().__init__()
        self.dist_basis = DistanceBasis(
            dist_feat_dim, cutoff=dist_feat_cutoff, envelope='nocusp'
        )
        self.schnet = (
            ElectronicSchNet(
                n_up,
                n_down,
                n_atoms,
                dist_feat_dim=dist_feat_dim,
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
                dist_feat_dim=dist_feat_dim,
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
        edges_nuc = self.dist_basis(dists_nuc)
        embeddings = {}
        if self.mf_schnet:
            embeddings['mean-field'] = self.mf_schnet(edges_nuc)
        if self.schnet:
            edges_elec = self.dist_basis(dists_elec)
            embeddings['many-body'] = self.schnet(edges_elec, edges_nuc)
        jastrow = (
            self.jastrow(embeddings[self.jastrow_type]) if self.jastrow_type else None
        )
        backflow = (
            self.backflow(embeddings[self.backflow_type])
            if self.backflow_type
            else None
        )
        return jastrow, backflow
