# DeepQMC

[![code style](https://img.shields.io/badge/code%20style-black-202020.svg)](https://github.com/ambv/black)

DeepQMC implements variational quantum Monte Carlo for electrons in molecules, using deep neural networks written in [PyTorch](https://pytorch.org) as trial wave functions. Besides the core functionality, it contains implementations of the following ansatzes:

- PauliNet: https://arxiv.org/abs/1909.08423

## Installing

Install and update using [Pip](https://pip.pypa.io/en/stable/quickstart/).

```
pip install -U deepqmc
```

## A simple example

```python
from deepqmc import Molecule, train, PauliNet

mol = Molecule.from_name('LiH')
net = PauliNet(mol)
train(net)
```

## Links

- Documentation: https://noegroup.github.io/deepqmc/
