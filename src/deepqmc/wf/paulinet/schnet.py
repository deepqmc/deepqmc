from itertools import product

import torch
from torch import nn

from deepqmc.torchext import SSP, get_log_dnn
from deepqmc.utils import NULL_DEBUG

from .indexing import pair_idxs, spin_pair_idxs

__version__ = '0.1.0'
__all__ = ['ElectronicSchNet']


class SubnetFactory:
    r"""Creates subnetworks for :class:`ElectronicSchNet`.

    The :class:`ElectronicSchNet` constructor expects this class constructor. To
    change the default subnetwork depths, use :func:`functools.partial`.

    Args:
        dist_feat_dim (int): :math:`\dim(\mathbf e)`, number of distance features
        kernel_dim (int): :math:`\dim(\mathbf w)`, dimension of the convolution kernel
        embedding_dim (int): :math:`\dim(\mathbf X)`, dimension of electron embeddings
        n_layers_w (int): number of layers in the :math:`\mathbf w` networks
        n_layers_h (int): number of layers in the :math:`\mathbf h` networks
        n_layers_g (int): number of layers in the :math:`\mathbf g` networks
    """

    def __init__(
        self,
        dist_feat_dim,
        kernel_dim,
        embedding_dim,
        *,
        n_layers_w=2,
        n_layers_h=1,
        n_layers_g=1,
    ):
        self.dist_feat_dim = dist_feat_dim
        self.kernel_dim = kernel_dim
        self.embedding_dim = embedding_dim
        self.n_layers_w = n_layers_w
        self.n_layers_h = n_layers_h
        self.n_layers_g = n_layers_g

    def w_subnet(self):
        r"""Create the :math:`\mathbf w` network."""
        return get_log_dnn(
            self.dist_feat_dim, self.kernel_dim, SSP, n_layers=self.n_layers_w
        )

    def h_subnet(self):
        r"""Create the :math:`\mathbf h` network."""
        return get_log_dnn(
            self.embedding_dim, self.kernel_dim, SSP, n_layers=self.n_layers_h,
        )

    def g_subnet(self):
        r"""Create the :math:`\mathbf g` network."""
        return get_log_dnn(
            self.kernel_dim, self.embedding_dim, SSP, n_layers=self.n_layers_g,
        )


class ElectronicSchNet(nn.Module):
    r"""Graph neural network SchNet adapted to handle electrons.

    SchNet constructs many-body feature representations of electrons that are
    invariant with respect to the translation and rotation of the whole molecule
    and equivariant with respect to exchange of same-spin electrons, by
    iteratively refining initial one-electron embeddings through mutual message
    passing modulated by electron distances,

    .. math::
        \begin{aligned}
        \mathbf x_i^{(0)}&:=\mathbf X \\
        \mathbf x_i^{(n+1)}&:=\mathbf x_i^{(n)}
          +\boldsymbol\chi^{(n)}_{\boldsymbol\theta}\big(
          \big\{\mathbf x_j^{(n)},\{\mathbf e(|\mathbf r_j-\mathbf r_k|)\}\big\}
          \big)
        \end{aligned}

    This module implements two versions of the iterative update rule, with only
    version 2 being documented,

    .. math::
        \begin{aligned}
        \mathbf z_i^{(n,\pm)}&:=\sum\nolimits_{j\neq i}^\pm
          \mathbf w^{(n,\pm)}_{\boldsymbol\theta}
          \big(\mathbf e(\lvert\mathbf r_i-\mathbf r_j\rvert)\big)
          \odot\mathbf h_{\boldsymbol\theta}^{(n)}\big(\mathbf x_j^{(n)}\big) \\
        \mathbf z_i^{(n,\mathrm n)}&:=\sum\nolimits_I
          \mathbf w_{\boldsymbol\theta}^{(n,\mathrm n)}
          \big(\mathbf e(\lvert\mathbf r_i-\mathbf R_I\rvert)\big)
          \odot\mathbf Y_{\boldsymbol\theta,I} \\
        \mathbf x_i^{(n+1)}&:=\mathbf x_i^{(n)}
          +\sum\nolimits_\pm\mathbf g^{(n,\pm)}_{\boldsymbol\theta}
          \big(\mathbf z_i^{(n,\pm)}\big)
          +\mathbf g^{(n,\mathrm n)}_{\boldsymbol\theta}
          \big(\mathbf z_i^{(n,\mathrm n)}\big)
        \end{aligned}


    Here, ":math:`\odot`" denotes element-wise multiplication,
    :math:`\mathbf w_{\boldsymbol\theta}^{(n)}`,
    :math:`\mathbf h^{(n)}_{\boldsymbol\theta}`, and
    :math:`\mathbf g_{\boldsymbol\theta}^{(n)}` are trainable functions
    represented by vanilla fully-connected (deep) neural networks,
    :math:`\mathbf Y_{\boldsymbol\theta},I}` are nuclear embeddings, and
    :math:`\mathbf e` are distance features.

    Args:
        n_up (int): :math:`N^\uparrow`, number of spin-up electrons
        n_down (int): :math:`N^\downarrow`, number of spin-down electrons
        n_nuclei (int): *M*, number of nuclei in a molecule
        embedding_dim (int): :math:`\dim(\mathbf X)`, dimension of electron embeddings
        dist_feat_dim (int): :math:`\dim(\mathbf e)`, number of distance features
        n_interactions (int): *L*, number of message passing iterations
        kernel_dim (int): :math:`\dim(\mathbf w)`, dimension of the convolution kernel
        subnet_metafactory (callable): factory,
            :math:`(\dim(\mathbf e),\dim(\mathbf w),\dim(\mathbf X))`
            :math:`\rightarrow(\mathbf w,\mathbf h,\mathbf g)`, with the
            interface of :class:`SubnetFactory`
        return_interactions (bool): whether calling the instance will return also
            intermediate electron embeddings, :math:`\mathbf x_i^{(n)}`, as the
            second output value
        version (int): architecture version, one of ``1`` or ``2``

    Shape:
        - Input1, :math:`\mathbf e(\lvert\mathbf r_i-\mathbf r_j\rvert)`:
          :math:`(*,N,N,\dim(\mathbf e))`
        - Input2, :math:`\mathbf e(\lvert\mathbf r_i-\mathbf R_I\rvert)`:
          :math:`(*,N,M,\dim(\mathbf e))`
        - Output: :math:`\mathbf x_i^{(L)}`: :math:`(*,N,\dim(\mathbf X))`

    Attributes:
        X: electronic embeddings of shape :math:`(1,\dim(\mathbf X))` if
            :math:`N^\uparrow=N^\downarrow` and :math:`(2,\dim(\mathbf X))`
            otherwise
        Y: nuclear embeddings of shape :math:`(M,\dim(\mathbf w))`
        w: :class:`torch.nn.ModuleDict` that contain all trainable
            :math:`\mathbf w^{(\cdot)}` subnetworks
        g: :class:`torch.nn.ModuleDict` that contain all trainable
            :math:`\mathbf g^{(\cdot)}` subnetworks
        h: :class:`torch.nn.ModuleDict` that contain all trainable
            :math:`\mathbf h^{(\cdot)}` subnetworks
    """

    def __init__(
        self,
        n_up,
        n_down,
        n_nuclei,
        embedding_dim,
        dist_feat_dim,
        subnet_metafactory=None,
        return_interactions=False,
        *,
        n_interactions=3,
        kernel_dim=64,
        version=2,
    ):
        assert version in {1, 2}
        subnet_metafactory = subnet_metafactory or SubnetFactory
        factory = subnet_metafactory(dist_feat_dim, kernel_dim, embedding_dim)
        super().__init__()
        self.version = version
        self.n_up, self.n_down = n_up, n_down
        self.n_interactions = n_interactions
        self.return_interactions = return_interactions
        self.Y = nn.Parameter(torch.randn(n_nuclei, kernel_dim))
        self.X = nn.Parameter(torch.randn(1 if n_up == n_down else 2, embedding_dim))
        w, h, g = {}, {}, {}
        for n in range(n_interactions):
            if version == 1:
                w[f'{n}'] = factory.w_subnet()
                g[f'{n}'] = factory.g_subnet()
            elif version == 2:
                for label in [True, False, 'n']:
                    w[f'{n},{label}'] = factory.w_subnet()
                    g[f'{n},{label}'] = factory.g_subnet()
            h[f'{n}'] = factory.h_subnet()
        self.w, self.h, self.g = map(nn.ModuleDict, [w, h, g])

    def forward(self, edges_elec, edges_nuc, debug=NULL_DEBUG):
        *batch_dims, n_elec = edges_nuc.shape[:-2]
        assert edges_elec.shape[:-1] == (*batch_dims, n_elec, n_elec)
        assert n_elec == self.n_up + self.n_down
        interactions = [] if self.return_interactions else None
        x = debug[0] = torch.cat(
            [
                X.clone().expand(n, -1)
                for X, n in zip(
                    self.X, [self.n_up, self.n_down] if len(self.X) == 2 else [n_elec]
                )
            ]
        ).expand(*batch_dims, -1, -1)
        for n in range(self.n_interactions):
            h = self.h[f'{n}'](x)
            if self.version == 1:
                i, j = pair_idxs(n_elec).T
                z_elec = (
                    (self.w[f'{n}'](edges_elec[..., i, j, :]) * h[..., j, :])
                    .view(*batch_dims, n_elec, -1, h.shape[-1])
                    .sum(dim=-2)
                )
                z_nuc = (self.w[f'{n}'](edges_nuc) * self.Y[..., None, :, :]).sum(
                    dim=-2
                )
                z = self.g[f'{n}'](z_elec + z_nuc)
            elif self.version == 2:
                z_elec_uu, z_elec_ud, z_elec_du, z_elec_dd = (
                    (self.w[f'{n},{si == sj}'](edges_elec[..., i, j, :]) * h[..., j, :])
                    .view(*batch_dims, n_si, -1, h.shape[-1])
                    .sum(dim=-2)
                    for (si, sj), (i, j), n_si in zip(
                        product('ud', repeat=2),
                        spin_pair_idxs(self.n_up, self.n_down, transposed=True),
                        [self.n_up, self.n_up, self.n_down, self.n_down],
                    )
                )
                z_elec_same = torch.cat([z_elec_uu, z_elec_dd], dim=-2)
                z_elec_anti = torch.cat([z_elec_ud, z_elec_du], dim=-2)
                z_nuc = (self.w[f'{n},n'](edges_nuc) * self.Y[..., None, :, :]).sum(
                    dim=-2
                )
                z = (
                    self.g[f'{n},{True}'](z_elec_same)
                    + self.g[f'{n},{False}'](z_elec_anti)
                    + self.g[f'{n},n'](z_nuc)
                )
            if interactions is not None:
                interactions.append(z)
            x = debug[n + 1] = x + z
        return x if interactions is None else interactions
