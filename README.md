# DeepQMC

![checks](https://img.shields.io/github/checks-status/deepqmc/deepqmc/master.svg)
[![coverage](https://img.shields.io/codecov/c/github/deepqmc/deepqmc.svg)](https://codecov.io/gh/deepqmc/deepqmc)
![python](https://img.shields.io/pypi/pyversions/deepqmc.svg)
[![pypi](https://img.shields.io/pypi/v/deepqmc.svg)](https://pypi.org/project/deepqmc/)
[![commits since](https://img.shields.io/github/commits-since/deepqmc/deepqmc/latest.svg)](https://github.com/deepqmc/deepqmc/releases)
[![last commit](https://img.shields.io/github/last-commit/deepqmc/deepqmc.svg)](https://github.com/deepqmc/deepqmc/commits/master)
[![license](https://img.shields.io/github/license/deepqmc/deepqmc.svg)](https://github.com/deepqmc/deepqmc/blob/master/LICENSE)
[![code style](https://img.shields.io/badge/code%20style-black-202020.svg)](https://github.com/ambv/black)
[![chat](https://img.shields.io/gitter/room/deepqmc/community)](https://gitter.im/deepqmc/community)
[![doi](https://img.shields.io/badge/doi-10.5281%2Fzenodo.3960826-blue)](http://doi.org/10.5281/zenodo.3960826)

DeepQMC implements variational quantum Monte Carlo for electrons in molecules, using deep neural networks written in [PyTorch](https://pytorch.org) as trial wave functions. Besides the core functionality, it contains implementations of the following ansatzes:

- PauliNet: https://doi.org/ghcm5p

## Installing

Install and update using [Pip](https://pip.pypa.io/en/stable/quickstart/).

```
pip install -U deepqmc[wf,train,cli]
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

Or on the command line:

```
$ cat lih/param.toml
system = 'LiH'
ansatz = 'paulinet'
[train_kwargs]
n_steps = 40
$ deepqmc train lih --save-every 20
converged SCF energy = -7.9846409186467
equilibrating: 49it [00:07,  6.62it/s]
training: 100%|███████| 40/40 [01:30<00:00,  2.27s/it, E=-8.0302(29)]
$ ln -s chkpts/state-00040.pt lih/state.pt
$ deepqmc evaluate lih
evaluating:  24%|▋  | 136/565 [01:12<03:40,  1.65it/s, E=-8.0396(17)]
```

## Links

- Documentation: https://deepqmc.github.io
