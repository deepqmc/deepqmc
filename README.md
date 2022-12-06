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

DeepQMC implements variational quantum Monte Carlo for electrons in molecules, using deep neural networks as trial wave functions. The package is based on [JAX](https://github.com/google/jax) and [Haiku](https://github.com/deepmind/dm-haiku). Besides the core functionality, it contains an implementation of the [PauliNet](https://doi.org/ghcm5p) ansatz.

### Installing

Install and update using [Pip](https://pip.pypa.io/en/stable/quickstart/):

```
pip install -U deepqmc -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

To install DeepQMC from a local Git repository run:

```
git clone https://github.com/deepqmc/deepqmc
cd deepqmc
pip install -e .[dev] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Documentation and exemplary usage

For further information about the DeepQMC package and tutorials covering the basic usage visit the [documentation](https://deepqmc.github.io).
