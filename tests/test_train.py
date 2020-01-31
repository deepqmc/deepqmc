from deepqmc import Molecule, train
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
