from functools import lru_cache, partial

import torch
from torch import nn

from deepqmc.torchext import SSP, fp_tensor, get_mlp, idx_perm

from .distbasis import DistanceBasis, DifferenceBasis

__version__ = '0.2.0'
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
        w_subnet=None,
        h_subnet=None,
        g_subnet=None,
        *,
        n_layers_w=2,
        n_layers_h=1,
        n_layers_g=1,
        **kwargs,
    ):
        kwargs.setdefault('activation', SSP)
        self.dist_feat_dim = dist_feat_dim
        self.kernel_dim = kernel_dim
        self.embedding_dim = embedding_dim
        self.kwargs_w = {
            **kwargs,
            **(w_subnet or {'hidden_layers': ('log', n_layers_w)}),
        }
        self.kwargs_h = {
            **kwargs,
            **(h_subnet or {'hidden_layers': ('log', n_layers_h)}),
        }
        self.kwargs_g = {
            **kwargs,
            **(g_subnet or {'hidden_layers': ('log', n_layers_g)}),
        }

    def w_subnet(self):
        r"""Create the :math:`\mathbf w` network."""
        return get_mlp(self.dist_feat_dim, self.kernel_dim, **self.kwargs_w)

    def h_subnet(self):
        r"""Create the :math:`\mathbf h` network."""
        return get_mlp(self.embedding_dim, self.kernel_dim, **self.kwargs_h)

    def g_subnet(self):
        r"""Create the :math:`\mathbf g` network."""
        return get_mlp(self.kernel_dim, self.embedding_dim, **self.kwargs_g)


class SchNetLayer(nn.Module):
    def __init__(self, factory, n_up):
        super().__init__()
        self.w = factory.w_subnet()
        self.g = factory.g_subnet()
        self.h = factory.h_subnet()

    def forward(self, x, Y, edges_elec, edges_nuc):
        *batch_dims, n_elec = edges_nuc.shape[:-2]
        h = self.h(x)
        i, j = idx_perm(n_elec, 2, x.device)
        z_elec = (self.w(edges_elec[..., i, j, :]) * h[..., j, :]).sum(dim=-2)
        z_nuc = (self.w(edges_nuc) * Y[..., None, :, :]).sum(dim=-2)
        return self.g(z_elec + z_nuc), None


@lru_cache()
def idx_pair_spin(n_up, n_down, device=torch.device('cpu')):  # noqa: B008
    # indexes for up-up, up-down, down-up, down-down
    ij = idx_perm(n_up + n_down, 2, device=device)
    mask = ij < n_up
    return [
        ('same', ij[:, mask[0] & mask[1]].view(2, n_up, max(n_up - 1, 0))),
        ('anti', ij[:, mask[0] & ~mask[1]].view(2, n_up, n_down)),
        ('anti', ij[:, ~mask[0] & mask[1]].view(2, n_down, n_up)),
        ('same', ij[:, ~mask[0] & ~mask[1]].view(2, n_down, max(n_down - 1, 0))),
    ]


class SchNetSpinLayer(nn.Module):
    def __init__(self, factory, n_up, shared_g=False, shared_h=True):
        super().__init__()
        self.shared_g = shared_g
        self.shared_h = shared_h
        labels = ['same', 'anti', 'n']
        self.w = nn.ModuleDict((lbl, factory.w_subnet()) for lbl in labels)
        self.g = (
            factory.g_subnet()
            if shared_g
            else nn.ModuleDict((lbl, factory.g_subnet()) for lbl in labels)
        )
        self.h = (
            factory.h_subnet()
            if shared_h
            else nn.ModuleDict((lbl, factory.h_subnet()) for lbl in labels[:2])
        )
        self.n_up = n_up

    def forward(self, x, Y, edges_elec, edges_nuc):
        *batch_dims, n_elec = edges_nuc.shape[:-2]
        n_up, n_down = self.n_up, n_elec - self.n_up
        messages_el = [
            (
                self.w[l](edges_elec[..., i, j, :])
                * (self.h if self.shared_h else self.h[l])(x)[..., j, :]
            )
            for l, (i, j) in idx_pair_spin(n_up, n_down, x.device)
        ]
        z_elec_uu, z_elec_ud, z_elec_du, z_elec_dd = (
            m.sum(dim=-2) for m in messages_el
        )
        messages_el = torch.cat(
            [
                torch.cat(messages_el[idx], dim=-2)
                for idx in (slice(None, 2), slice(2, None))
            ],
            dim=-3,
        )
        z_elec_same = torch.cat([z_elec_uu, z_elec_dd], dim=-2)
        z_elec_anti = torch.cat([z_elec_ud, z_elec_du], dim=-2)
        messages_nuc = self.w['n'](edges_nuc) * Y[..., None, :, :]
        z_nuc = messages_nuc.sum(dim=-2)
        update = (
            self.g(z_elec_same + z_elec_anti + z_nuc)
            if self.shared_g
            else self.g['same'](z_elec_same)
            + self.g['anti'](z_elec_anti)
            + self.g['n'](z_nuc)
        )
        return update, (messages_el, messages_nuc)


class NuclearEmbedding(nn.Module):
    def __init__(self, n_nuclei, kernel_dim, charges, **kwargs):
        assert len(charges) == n_nuclei
        super().__init__()
        self.mlp = get_mlp(1, kernel_dim, **kwargs)
        self.register_buffer('charges', fp_tensor(charges))

    def forward(self, idxs):
        return self.mlp(self.charges[idxs].unsqueeze(dim=-1))


class ElectronEmbedding(nn.Module):
    def __init__(self, n_nuclei, embedding_dim, n_layers=2, activation=SSP, **kwargs):
        super().__init__()
        self.mlp = get_mlp(
            4 * n_nuclei, embedding_dim, ('log', n_layers), activation, **kwargs
        )

    def forward(self, dists, diffs):
        return self.mlp(torch.cat([dists, diffs.flatten(start_dim=-2)], dim=-1))


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
    :math:`\mathbf Y_{\boldsymbol\theta,I}` are nuclear embeddings, and
    :math:`\mathbf e` are distance features.

    Args:
        n_up (int): :math:`N^\uparrow`, number of spin-up electrons
        n_down (int): :math:`N^\downarrow`, number of spin-down electrons
        n_nuclei (int): *M*, number of nuclei in a molecule
        embedding_dim (int): :math:`\dim(\mathbf X)`, dimension of electron embeddings
        subnet_metafactory (callable): factory,
            :math:`(\dim(\mathbf e),\dim(\mathbf w),\dim(\mathbf X))`
            :math:`\rightarrow(\mathbf w,\mathbf h,\mathbf g)`, with the
            interface of :class:`SubnetFactory`
        dist_feat_dim (int): :math:`\dim(\mathbf e)`, number of distance features
        dist_feat_cutoff (float, a.u.): distance at which distance features
            go to zero
        n_interactions (int): *L*, number of message passing iterations
        kernel_dim (int): :math:`\dim(\mathbf w)`, dimension of the convolution kernel
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

    LAYER_FACTORIES = {
        1: SchNetLayer,
        2: SchNetSpinLayer,
    }

    def __init__(
        self,
        n_up,
        n_down,
        n_nuclei,
        embedding_dim,
        subnet_metafactory=None,
        dist_basis=None,
        layer_kwargs=None,
        nuc_embedding=nn.Embedding,
        init_embed_one=False,
        resnet=True,
        *,
        dist_feat_dim=32,
        dist_feat_cutoff=10.0,
        n_interactions=3,
        kernel_dim=64,
        version=2,
        layer_norm=False,
        use_diffs=False,
        initial_embedding='embedding',
    ):
        assert version in self.LAYER_FACTORIES
        assert initial_embedding in ['embedding', 'distance']
        self.initial_embedding = initial_embedding
        subnet_metafactory = subnet_metafactory or SubnetFactory
        if not dist_basis:
            dist_basis = partial(DistanceBasis, envelope='nocusp')
            diff_basis = partial(DifferenceBasis, envelope='nocusp')
        self.use_diffs = use_diffs
        super().__init__()
        if self.use_diffs:
            total_dist_feat_dim = 7 * dist_feat_dim  # distance + difference
            self.diff_basis = diff_basis(dist_feat_dim, dist_feat_cutoff)
        else:
            total_dist_feat_dim = dist_feat_dim
        subnet_factory = subnet_metafactory(
            total_dist_feat_dim, kernel_dim, embedding_dim
        )
        self.resnet = resnet
        self.kernel_dim = kernel_dim
        self.dist_basis = dist_basis(dist_feat_dim, dist_feat_cutoff)
        self.Y = nuc_embedding(n_nuclei, kernel_dim)
        if initial_embedding == 'embedding':
            self.X = nn.Embedding(1 if n_up == n_down else 2, embedding_dim)
        elif initial_embedding == 'distance':
            self.X = ElectronEmbedding(n_nuclei, embedding_dim)
        if init_embed_one and initial_embedding == 'embedding':
            self.X.weight.detach().fill_(1.0)
        self.layers = nn.ModuleList(
            self.LAYER_FACTORIES[version](subnet_factory, n_up, **(layer_kwargs or {}))
            for _ in range(n_interactions)
        )
        self.layer_norms = (
            nn.ModuleList(nn.LayerNorm(embedding_dim) for _ in range(n_interactions))
            if layer_norm
            else [None for _ in range(n_interactions)]
        )
        spin_idxs = torch.tensor(
            (n_up + n_down) * [0] if n_up == n_down else n_up * [0] + n_down * [1]
        )
        self.register_buffer('spin_idxs', spin_idxs)
        self.register_buffer('nuclei_idxs', torch.arange(n_nuclei))

    def forward(self, dists_elec, dists_nuc, diffs_elec=None, diffs_nuc=None):
        *batch_dims, n_elec, n_nuclei = dists_nuc.shape
        assert dists_elec.shape == (*batch_dims, n_elec, n_elec)
        assert n_elec == len(self.spin_idxs)
        edges_nuc = self.dist_basis(dists_nuc)
        edges_elec = self.dist_basis(dists_elec)
        if self.use_diffs:
            edges_nuc = torch.cat([edges_nuc, self.diff_basis(diffs_nuc)], dim=-1)
            edges_elec = torch.cat([edges_elec, self.diff_basis(diffs_elec)], dim=-1)
        if self.initial_embedding == 'embedding':
            x = self.X(self.spin_idxs.expand(*batch_dims, -1))
        elif self.initial_embedding == 'distance':
            x = self.X(dists_nuc, diffs_nuc)
        Y = self.Y(self.nuclei_idxs.expand(*batch_dims, -1))
        for layer, norm in zip(self.layers, self.layer_norms):
            z, messages = layer(x, Y, edges_elec, edges_nuc)
            if norm:
                z = 0.1 * norm(z)
            x = x + z if self.resnet else z
        return x, messages
