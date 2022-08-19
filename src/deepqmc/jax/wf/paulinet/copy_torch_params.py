import jax.numpy as jnp
from torch import nn


def get_embedding_params(torch_embed):
    return {'w': jnp.array(torch_embed.weight.detach())}


def get_linear_params(torch_linear):
    w = jnp.array(torch_linear.weight.detach().T)
    if torch_linear.bias is None:
        return {'w': w}
    else:
        b = jnp.array(torch_linear.bias.detach())
        return {'w': w, 'b': b}


def get_mlp_params(torch_mlp, prefix):
    params = {}
    for i, layer in enumerate(filter(lambda x: isinstance(x, nn.Linear), torch_mlp)):
        params[prefix + f'linear_{i}'] = get_linear_params(layer)
    return params


def get_ee_params(torch_ee):
    zetas = []
    for shell in torch_ee.shells:
        zetas.append(*shell.zetas.detach().tolist())
    return {'zetas': jnp.array(zetas)}


def get_schnet_params(torch_schnet, prefix):
    params = {}
    prefix += 'SchNet/~/'
    labels = ['_ne', '_same', '_anti']
    params[prefix + 'NuclearEmbedding'] = get_embedding_params(torch_schnet.Y)
    params[prefix + 'ElectronicEmbedding'] = get_embedding_params(torch_schnet.X)
    prefix += 'SchNetLayer_'
    for i, layer in enumerate(torch_schnet.layers):
        for sub_label in ['w', 'h', 'g']:
            if sub_label == 'w':
                channels = labels
            elif sub_label == 'h':
                if layer.shared_h:
                    channels = ['']
                else:
                    channels = labels[1:]
            elif sub_label == 'g':
                if layer.shared_g:
                    channels = ['']
                else:
                    channels = labels
            subnet = getattr(layer, sub_label)
            for channel in channels:
                paulinet_channel = channel
                if channel == '_ne':
                    paulinet_channel = '_n'
                mlp_params = get_mlp_params(
                    subnet[paulinet_channel[1:]] if channel else subnet,
                    prefix + f'{i}/~/' + sub_label + channel + '/~/',
                )
                params = dict(params, **mlp_params)
    return params


def get_omni_params(torch_omni, prefix):
    if torch_omni.schnet is None:
        return {}
    params = {}
    prefix += 'omni_net/~/'
    sep = '/~/'
    schnet_params = get_schnet_params(torch_omni.schnet, prefix)
    params = dict(params, **schnet_params)
    if getattr(torch_omni, 'jastrow', False):
        jastrow_params = get_mlp_params(
            torch_omni.jastrow.net, prefix + 'Jastrow' + sep + 'mlp' + sep
        )
        params = dict(params, **jastrow_params)

    if getattr(torch_omni, 'backflow', False):

        if getattr(torch_omni.backflow, 'up', False):
            backflows = [
                ('Backflow_up', torch_omni.backflow.up),
                ('Backflow_down', torch_omni.backflow.down),
            ]

        else:
            backflows = [('Backflow', torch_omni.backflow)]

        for name, backflow in backflows:

            for i, mlp in enumerate(backflow.mlps):
                if i == 0:
                    mlp_lbl = ''
                else:
                    mlp_lbl = f'_{i}'

                backflow_params = get_mlp_params(
                    mlp, prefix + name + sep + 'mlp' + mlp_lbl + sep
                )

            params = dict(params, **backflow_params)
    return params


def get_paulinet_params(torch_paulinet):
    params = {}
    prefix = 'pauli_net/~/'
    params[prefix + 'exponential_envelopes'] = get_ee_params(torch_paulinet.mo.basis)
    params[prefix + 'mo_coeff'] = get_linear_params(torch_paulinet.mo.mo_coeff)
    if isinstance(torch_paulinet.conf_coeff, nn.Identity):
        params[prefix + 'conf_coeff'] = {'w': jnp.eye(len(torch_paulinet.confs))}
    else:
        params[prefix + 'conf_coeff'] = get_linear_params(torch_paulinet.conf_coeff)
    omni_params = get_omni_params(torch_paulinet.omni, prefix)
    params = dict(params, **omni_params)
    return params
