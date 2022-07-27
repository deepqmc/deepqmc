#!/usr/bin/env python3

import haiku as hk
import jax.numpy as jnp
import torch
from jax import random, vmap
from schnet import SchNet, SchNetEdgesBuilder
from torch import nn

from deepqmc import Molecule
from deepqmc.jax.molecule import Molecule as jMolecule
from deepqmc.physics import pairwise_distance, pairwise_self_distance
from deepqmc.wf.paulinet.schnet import ElectronicSchNet


def get_linear_params(torch_lin):
    w = jnp.array(torch_lin.weight.detach().T)
    try:
        b = jnp.array(torch_lin.bias.detach())
    except Exception:
        b = None
    if b is None:
        return {'w': w}
    else:
        return {'w': w, 'b': b}


def get_mlp_params(torch_mlp, prefix):
    params = {}
    for i, layer in enumerate(filter(lambda x: isinstance(x, nn.Linear), torch_mlp)):
        params[prefix + '/~/' + f'linear_{i}'] = get_linear_params(layer)
    return params


def get_embedding_params(torch_embed):
    return {'embeddings': jnp.array(torch_embed.weight.detach())}


def get_schnet_params(torch_schnet):
    params = {}
    prefix = 'SchNet/~/'
    sep = '/~/'
    labels = ['_n', '_same', '_anti']
    params[prefix + 'NuclearEmbedding'] = get_embedding_params(schnet.Y)
    params[prefix + 'ElectronicEmbedding'] = get_embedding_params(schnet.X)
    prefix += 'SchNetLayer_'
    for i, layer in enumerate(torch_schnet.layers):
        for sub_label in ['w', 'h', 'g']:
            if sub_label == 'w':
                channels = labels
            elif sub_label == 'h':
                if layer.shared_h:
                    channels = ['']
                else:
                    channels = labels
            elif sub_label == 'g':
                if layer.shared_g:
                    channels = ['']
                else:
                    channels = labels
            subnet = getattr(layer, sub_label)
            for channel in channels:
                mlp_params = get_mlp_params(
                    subnet[channel[1:]] if channel else subnet,
                    prefix + f'{i}' + sep + sub_label + channel,
                )
                params = dict(params, **mlp_params)
    return params


if __name__ == "__main__":
    rng_schnet = random.PRNGKey(0)

    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]
    charges = [10, 20]
    nbatch = 100
    embedding_dim = 64
    dist_feat_dim = 32
    kernel_dim = 64
    n_interactions = 3
    layer_kwargs = {'n_layers_w': 2, 'n_layers_h': 1, 'n_layers_g': 1}

    jmol = jMolecule(coords, charges, 0, 0)
    mol = Molecule(torch.tensor(coords), torch.tensor(charges), 0, 0)
    n_elec = int(mol.charges.sum()) - mol.charge
    n_nuc = len(mol.charges)
    n_up = (n_elec + mol.spin) // 2
    n_down = n_elec - n_up
    rs = torch.rand(n_elec, 3)
    jrs = jnp.array(rs)
    rs_batch = torch.rand(nbatch, n_elec, 3)
    jrs_batch = jnp.array(rs_batch)

    schnet = ElectronicSchNet(
        n_up,
        n_down,
        n_nuc,
        embedding_dim,
        dist_feat_dim=dist_feat_dim,
        n_interactions=n_interactions,
        kernel_dim=kernel_dim,
        subnet_kwargs=layer_kwargs,
    )

    schnet_edges_builder = SchNetEdgesBuilder(jmol)

    def _schnet(rs, edges):
        return SchNet(
            n_nuc,
            n_up,
            n_down,
            jmol.coords,
            embedding_dim,
            dist_feat_dim,
            kernel_dim,
            n_interactions=n_interactions,
            layer_kwargs=layer_kwargs,
        )(rs, edges)

    jschnet = hk.without_apply_rng(hk.transform(_schnet))
    edges = schnet_edges_builder(jrs)
    edges_batch = schnet_edges_builder(jrs_batch)
    params = jschnet.init(rng_schnet, jrs, edges)
    vschnet = vmap(jschnet.apply, (None, 0, 0))
    params = get_schnet_params(schnet)
    dists_elec = pairwise_self_distance(rs_batch, full=True)
    dists_nuc = pairwise_distance(rs_batch, mol.coords)
    torch_result = jnp.array(schnet(dists_elec, dists_nuc)[0].detach())
    jax_result = vschnet(params, jrs_batch, edges_batch)

    print(
        'difference:'
        f' {jnp.abs(torch_result - jax_result).sum() / len(jax_result.reshape(-1))}'
    )
