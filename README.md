# DeepQMC

[![build](https://img.shields.io/travis/com/deepqmc/deepqmc/master.svg)](https://travis-ci.com/deepqmc/deepqmc)
[![coverage](https://img.shields.io/codecov/c/github/deepqmc/deepqmc.svg)](https://codecov.io/gh/deepqmc/deepqmc)
![python](https://img.shields.io/pypi/pyversions/deepqmc.svg)
[![pypi](https://img.shields.io/pypi/v/deepqmc.svg)](https://pypi.org/project/deepqmc/)
[![commits since](https://img.shields.io/github/commits-since/deepqmc/deepqmc/latest.svg)](https://github.com/deepqmc/deepqmc/releases)
[![last commit](https://img.shields.io/github/last-commit/deepqmc/deepqmc.svg)](https://github.com/deepqmc/deepqmc/commits/master)
[![license](https://img.shields.io/github/license/deepqmc/deepqmc.svg)](https://github.com/deepqmc/deepqmc/blob/master/LICENSE)
[![code style](https://img.shields.io/badge/code%20style-black-202020.svg)](https://github.com/ambv/black)
[![chat](https://img.shields.io/gitter/room/deepqmc/deepqmc)](https://gitter.im/deepqmc/deepqmc)
[![doi](https://img.shields.io/badge/doi-10.5281%2Fzenodo.3960826-blue)](http://doi.org/10.5281/zenodo.3960826)

DeepQMC implements variational quantum Monte Carlo for electrons in molecules, using deep neural networks written in [PyTorch](https://pytorch.org) as trial wave functions. Besides the core functionality, it contains implementations of the following ansatzes:

- PauliNet [[video](https://youtu.be/_bdvpmleAgw)]: https://arxiv.org/abs/1909.08423

## Installing

Install and update using [Pip](https://pip.pypa.io/en/stable/quickstart/).

```
pip install -U deepqmc[wf,train]
```

## A simple example

```python
>>> from deepqmc import Molecule, evaluate, train
>>> from deepqmc.wf import PauliNet
>>> mol = Molecule.from_name('LiH')
>>> net = PauliNet.from_hf(mol).cuda()
converged SCF energy = -7.9846409186467
>>> train(net)
equilibrating: 64it [00:08,  7.58it/s]
training:   0%|  | 46/10000 [01:37<5:50:59,  2.12s/it, E=-8.0371(24)]
KeyboardInterrupt
>>> evaluate(net)
evaluating:  23%|▋  | 134/571 [01:08<03:44,  1.94it/s, E=-8.0455(32)]
```

## Links

- Documentation: https://deepqmc.github.io
