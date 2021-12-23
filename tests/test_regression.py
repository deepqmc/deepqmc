from pathlib import Path

import pytest
import torch

from deepqmc import Molecule
from deepqmc.app import ansatz_from_name
from deepqmc.sampling import rand_from_mol


@pytest.fixture
def mol():
    return Molecule.from_name('LiH')


@pytest.fixture
def rs(mol):
    torch.manual_seed(0)
    return rand_from_mol(mol, 1, torch.tensor([1, -1]))


@pytest.mark.parametrize(
    'ansatz,kwargs',
    [
        ('paulinet', {}),
        ('paulinet', {'cas': (2, 2)}),
        ('paulinet', {'cas': (2, 2), 'conf_cutoff': 1}),
        ('paulinet', {'cas': (2, 2), 'conf_limit': 2}),
        ('paulinet', {'cas': (2, 2), 'backflow_type': 'det'}),
        ('paulinet', {'basis': '6-31g'}),
        ('paulinet', {'init_weights': False}),
        ('paulinet', {'freeze_mos': False}),
        ('paulinet', {'cusp_correction': False}),
        ('paulinet', {'cusp_electrons': False}),
        ('paulinet', {'backflow_channels': 2}),
        ('paulinet', {'backflow_transform': 'add'}),
        ('paulinet', {'backflow_transform': 'both'}),
        ('paulinet', {'rc_scaling': 5}),
        ('paulinet', {'cusp_alpha': 1.0}),
        ('paulinet', {'freeze_embed': True}),
        ('paulinet', {'omni_kwargs.omni_schnet.embedding_dim': 64}),
        ('paulinet', {'omni_kwargs.omni_schnet.jastrow': 'mean-field'}),
        ('paulinet', {'omni_kwargs.omni_schnet.backflow': 'mean-field'}),
        ('paulinet', {'omni_kwargs.omni_schnet.jastrow_kwargs.n_layers': 2}),
        ('paulinet', {'omni_kwargs.omni_schnet.jastrow_kwargs.sum_first': False}),
        ('paulinet', {'omni_kwargs.omni_schnet.backflow_kwargs.n_layers': 2}),
        ('paulinet', {'omni_kwargs.omni_schnet.schnet_kwargs.dist_feat_dim': 16}),
        ('paulinet', {'omni_kwargs.omni_schnet.schnet_kwargs.dist_feat_cutoff': 3.0}),
        ('paulinet', {'omni_kwargs.omni_schnet.schnet_kwargs.n_interactions': 2}),
        ('paulinet', {'omni_kwargs.omni_schnet.schnet_kwargs.kernel_dim': 32}),
        ('paulinet', {'omni_kwargs.omni_schnet.subnet_kwargs.n_layers_w': 3}),
        ('paulinet', {'omni_kwargs.omni_schnet.subnet_kwargs.n_layers_h': 2}),
        ('paulinet', {'omni_kwargs.omni_schnet.subnet_kwargs.n_layers_g': 2}),
    ],
    ids=lambda x: ','.join(f'{k}={v}' for k, v in x.items())
    if isinstance(x, dict)
    else x,
)
def test(ansatz, kwargs, mol, rs, num_regression, request):
    kwargs = {
        k.replace('omni_kwargs.omni_schnet', 'omni_factory')
        .replace('schnet_kwargs', 'schnet_factory')
        .replace('jastrow_kwargs', 'jastrow_factory')
        .replace('backflow_kwargs', 'backflow_factory')
        .replace('subnet_kwargs', 'schnet_factory.subnet_metafactory'): v
        for k, v in kwargs.items()
    }
    workdir = (
        Path(request.fspath.dirname)
        / 'workdirs'
        / (
            'with-cas'
            if 'cas' in kwargs
            else kwargs['basis']
            if 'basis' in kwargs
            else 'default'
        )
    )
    wf = ansatz_from_name(ansatz, mol, workdir=str(workdir), **kwargs)
    torch.manual_seed(0)
    for _, p in sorted(wf.named_parameters()):
        if p.requires_grad:
            torch.nn.init.normal_(p, std=1e-1)
    (psi,), (sign,) = wf(rs)
    psi.backward()
    grad_norm = torch.stack(
        [p.grad.norm() for p in wf.parameters() if p.grad is not None]
    ).norm()
    num_regression.check(
        {'psi': psi.detach(), 'sign': sign.detach(), 'grad_norm': grad_norm}
    )
