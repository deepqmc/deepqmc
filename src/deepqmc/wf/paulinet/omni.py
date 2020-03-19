from functools import partial

import torch
from torch import nn

from deepqmc.torchext import SSP, get_log_dnn
from deepqmc.utils import NULL_DEBUG

from .backflow import Backflow
from .schnet import ElectronicSchNet, SubnetFactory

__version__ = '0.2.0'
__all__ = ['OmniSchNet']


class OmniSchNet(nn.Module):
    r"""Combined Jastrow/backflow neural network based on SchNet.

    This module uses a single :class:`ElectronicSchNet` instance to build
    many-body feature representations of electrons, which are subsequently
    passed as an input into additional trainable functions to obtain the Jastrow
    factor and backflow transformations. This enables the use of a single SchNet
    module within :class:`~deepqmc.wf.PauliNet`, which itself makes no
    assumptions about parameter sharing by the Jastrow factor and backflow
    network modules.

    The Jastrow factor and backflowed orbitals are obtained as

    .. math::
        J:=\eta_{\boldsymbol\theta}\big(\textstyle\sum_i\mathbf x_i^{(L)}\big),\qquad
        \tilde\varphi_{\mu i}(\mathbf r)
        :=\bigg(1+2\tanh\Big(\kappa_{\boldsymbol\theta,\mu}
        \big(\mathbf x_i^{(L)}\big)\!\Big)\!\bigg)\varphi_\mu(\mathbf r_i)

    where :math:`\eta_{\boldsymbol\theta}` and
    :math:`\boldsymbol\kappa_{\boldsymbol\theta}` are vanilla deep neural networks.

    The Jastrow and backflow are obtained by calling :meth:`forward_jastrow` and
    :meth:`forward_backflow`, which calls SchNet only once under the hood. After
    the forward pass is finished, :math:`close` must be called, which ensures
    that SchNet is called a new in the next pass.

    Args:
        mol (:class:`~deepqmc.Molecule`): molecule whose wave function is represented
        dist_feat_dim (int): passed to :class:`ElectronicSchNet`
        n_up (int): passed to :class:`ElectronicSchNet`
        n_down (int): passed to :class:`ElectronicSchNet`
        n_orbitals (int): :math:`N_\text{orb}`, number of molecular orbitals
        embedding_dim (int): :math:`\dim(\mathbf x_i^{(L)})`, dimension of SchNet
            embeddings
        with_jastrow (bool): if false, :meth:``
        n_jastrow_layers (int): number of layers in the Jastrow factor network
        with_backflow (bool): if false,
            :math:`\tilde\varphi_{\mu i}(\mathbf r):=\varphi_\mu(\mathbf r_i)`
        n_backflow_layers (int): number of layers in the backflow network
        with_r_backflow (bool): whether real-space backflow is used
        schnet_kwargs (dict): extra arguments passed to :class:`ElectronicSchNet`
        subnet_kwargs (dict): extra arguments passed to :class:`SubnetFactory`

    Shape:
        - :meth:`forward_jastrow`:
            - Input1, :math:`\mathbf e(\lvert\mathbf r_i-\mathbf r_j\rvert)`:
              :math:`(*,N,N,\dim(\mathbf e))`
            - Input2, :math:`\mathbf e(\lvert\mathbf r_i-\mathbf R_I\rvert)`:
              :math:`(*,N,M,\dim(\mathbf e))`
            - Output, :math:`J`: :math:`(*)`
        - :meth:`forward_backflow`:
            - Input1, :math:`\varphi_\mu(\mathbf r_i)`: :math:`(*,N,N_\text{orb})`
            - Input2, :math:`\mathbf e(\lvert\mathbf r_i-\mathbf r_j\rvert)`:
              :math:`(*,N,N,\dim(\mathbf e))`
            - Input3, :math:`\mathbf e(\lvert\mathbf r_i-\mathbf R_I\rvert)`:
              :math:`(*,N,M,\dim(\mathbf e))`
            - Output, :math:`\tilde\varphi_{\mu i}(\mathbf r)`:
              :math:`(*,N,N_\text{orb})`

    Attributes:
        schnet: :class:`ElectronicSchNet` network
        jastrow: :class:`torch.nn.Sequential` representing a vanilla DNN
        backflow: :class:`torch.nn.Sequential` representing a vanilla DNN
    """

    def __init__(
        self,
        mol,
        dist_feat_dim,
        n_up,
        n_down,
        n_orbitals,
        *,
        embedding_dim=128,
        with_jastrow=True,
        n_jastrow_layers=3,
        with_backflow=True,
        n_backflow_layers=3,
        with_r_backflow=False,
        schnet_kwargs=None,
        subnet_kwargs=None,
    ):
        super().__init__()
        self.schnet = ElectronicSchNet(
            n_up,
            n_down,
            len(mol),
            dist_feat_dim=dist_feat_dim,
            embedding_dim=embedding_dim,
            subnet_metafactory=partial(SubnetFactory, **(subnet_kwargs or {})),
            **(schnet_kwargs or {}),
        )
        if with_jastrow:
            self.jastrow = get_log_dnn(embedding_dim, 1, SSP, n_layers=n_jastrow_layers)
        else:
            self.forward_jastrow = None
        if with_backflow:
            self.backflow = get_log_dnn(
                embedding_dim, n_orbitals, SSP, n_layers=n_backflow_layers,
            )
        else:
            self.forward_backflow = None
        if with_r_backflow:
            self.r_backflow = Backflow(mol, embedding_dim)
        else:
            self.forward_r_backflow = None
        self._cache = {}

    def _get_embeddings(self, edges_elec, edges_nuc, debug):
        edges_id = id(edges_elec) + id(edges_nuc)
        if self._cache.get('dist_feats_id') != edges_id:
            self._cache['dist_feats_id'] = edges_id
            with debug.cd('schnet'):
                self._cache['embeddings'] = self.schnet(
                    edges_elec, edges_nuc, debug=debug
                )
        return self._cache['embeddings']

    def forward_jastrow(self, edges_elec, edges_nuc, debug=NULL_DEBUG):
        """Evaluate Jastrow factor."""
        xs = self._get_embeddings(edges_elec, edges_nuc, debug)
        J = self.jastrow(xs.sum(dim=-2)).squeeze(dim=-1)
        return debug.result(J)

    def forward_backflow(self, edges_elec, edges_nuc, debug=NULL_DEBUG):
        """Evaluate backflow."""
        xs = self._get_embeddings(edges_elec, edges_nuc, debug)
        xs = debug['backflow'] = self.backflow(xs)
        return (1 + 2 * torch.tanh(xs / 4))

    def forward_r_backflow(self, rs, edges_elec, edges_nuc, debug=NULL_DEBUG):
        xs = self._get_embeddings(edges_elec, edges_nuc, debug)
        return self.r_backflow(rs, xs)

    def forward_close(self):
        """Clear the cached SchNet embeddings."""
        self._cache.clear()
