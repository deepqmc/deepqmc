from deepqmc import Molecule, evaluate, train
from deepqmc.wf import PauliNet


def test_simple_example():
    mol = Molecule.from_name('LiH')
    net = PauliNet.from_hf(mol, cas=(4, 2))
    train(
        net,
        n_steps=2,
        batch_size=5,
        epoch_size=2,
        sampler_kwargs={'sample_size': 5, 'n_discard': 0},
    )
    evaluate(
        net,
        n_steps=1,
        sample_size=5,
        n_decorrelate=0,
        sample_kwargs={'equilibrate': False, 'block_size': 1},
    )
