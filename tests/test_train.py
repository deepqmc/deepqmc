from deepqmc import Molecule, evaluate, train
from deepqmc.wf import PauliNet


def test_simple_example(tmp_path):
    mol = Molecule.from_name('LiH')
    net = PauliNet.from_hf(mol, cas=(4, 2), conf_limit=2)
    chkpts = []
    train(
        net,
        n_steps=3,
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
