# DeepQMC

![python](https://img.shields.io/badge/python-3.7-blue)
![license](https://img.shields.io/badge/license-MIT-orange)
[![code style](https://img.shields.io/badge/code%20style-black-202020.svg)](https://github.com/ambv/black)

DeepQMC implements variational quantum Monte Carlo for electrons in molecules, using deep neural networks written in [PyTorch](https://pytorch.org) as trial wave functions. Besides the core functionality, it contains implementations of the following ansatzes:

- PauliNet: https://arxiv.org/abs/1909.08423

## Installing

Install and update using [Pip](https://pip.pypa.io/en/stable/quickstart/).

```
pip install -U git+https://github.com/deepqmc/deepqmc.git#egg=deepqmc[wf,train]
```

## A simple example

```python
from deepqmc import Molecule, evaluate, train
from deepqmc.wf import PauliNet

mol = Molecule.from_name('LiH')
net = PauliNet.from_hf(mol).cuda()
train(net)
evaluate(net)
```

## Links

- Documentation: https://noegroup.github.io/deepqmc/
