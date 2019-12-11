from deepqmc import Molecule, train
from deepqmc.wf import PauliNet


def test_simple_example():
    mol = Molecule.from_name('LiH')
    net = PauliNet.from_hf(mol, cas=(4, 2))
    train(
        net,
        cuda=False,
        n_steps=2,
        sampler_size=5,
        batched_sampler_kwargs={'batch_size': 5, 'n_discard': 0, 'sample_every': 2},
    )
