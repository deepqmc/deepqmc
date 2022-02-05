import copy

import pytest
import torch

from deepqmc import Molecule, evaluate, train
from deepqmc.wf import PauliNet


@pytest.mark.parametrize('with_resampling', [False, True])
def test_simple_example(tmp_path, with_resampling):
    mol = Molecule.from_name('LiH')
    net = PauliNet.from_hf(mol, cas=(4, 2), conf_limit=2)
    if with_resampling:
        resampling_kwargs = {
            'keep_walker_weights': True,
            'resampling_frequency': 1,
        }
    else:
        resampling_kwargs = {}
    chkpts = []
    train(
        net,
        n_steps=4,
        batch_size=5,
        save_every=2,
        epoch_size=3,
        equilibrate=1,
        chkpts=chkpts,
        workdir=tmp_path,
        fit_kwargs={'subbatch_size': 5},
        sampler_kwargs={
            'sample_size': 15,
            'n_discard': 0,
            'n_decorrelate': 0,
            'n_first_certain': 0,
            **resampling_kwargs,
        },
    )
    train(
        net,
        n_steps=1,
        batch_size=5,
        epoch_size=1,
        state=chkpts[-1][1],
        equilibrate=False,
        fit_kwargs={'subbatch_size': 5},
        sampler_kwargs={
            'sample_size': 5,
            'n_discard': 0,
            'n_decorrelate': 0,
            'n_first_certain': 0,
        },
    )
    evaluate(
        net,
        n_steps=1,
        sample_size=5,
        log_dict={},
        sample_kwargs={'equilibrate': 1, 'block_size': 1},
        sampler_kwargs={'n_decorrelate': 0, 'n_first_certain': 0},
    )
    evaluate(
        net,
        n_steps=1,
        workdir=tmp_path,
        sample_size=5,
        sample_kwargs={'equilibrate': False, 'block_size': 1},
        sampler_kwargs={'n_decorrelate': 0, 'n_first_certain': 0},
    )


def test_invariance_to_subbatch_size(tmp_path):
    batch_size = 1000

    # For determinism we work on CPU and compute the Hartree-Fock solution once.
    mol = Molecule.from_name('LiH')
    net = PauliNet.from_hf(mol, cas=(4, 2), conf_limit=2).cpu()

    state = copy.deepcopy(net.state_dict())
    params_orig = copy.deepcopy(list(net.parameters()))

    def get_total_gradient_norm(subbatch_size: int):
        torch.manual_seed(0)
        net.load_state_dict(state)

        train(
            net,
            n_steps=1,
            batch_size=batch_size,
            epoch_size=1,
            optimizer='SGD',  # Loss scale variance would be hidden with Adam.
            equilibrate=False,
            workdir=tmp_path,
            fit_kwargs={'subbatch_size': subbatch_size},
            sampler_kwargs={
                'sample_size': batch_size,
                'n_discard': 0,
                'n_decorrelate': 0,
                'n_first_certain': 0,
            },
        )

        params = list(net.parameters())
        return sum((p1 - p2).norm() for p1, p2 in zip(params, params_orig))

    grad_norm_1 = get_total_gradient_norm(subbatch_size=50)
    grad_norm_2 = get_total_gradient_norm(subbatch_size=1000)

    # We accept relative variation up to a factor of 5. If the loss/gradients
    # scaled with subbatch size instead, then the expected ratio would be 20.
    assert torch.isclose(grad_norm_1, grad_norm_2, rtol=5.0)
