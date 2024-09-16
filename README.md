# DeepQMC

![checks](https://img.shields.io/github/actions/workflow/status/deepqmc/deepqmc/tests.yaml?label=tests)
[![coverage](https://img.shields.io/codecov/c/github/deepqmc/deepqmc.svg)](https://codecov.io/gh/deepqmc/deepqmc)
![python](https://img.shields.io/pypi/pyversions/deepqmc.svg)
[![pypi](https://img.shields.io/pypi/v/deepqmc.svg)](https://pypi.org/project/deepqmc/)
[![commits since](https://img.shields.io/github/commits-since/deepqmc/deepqmc/latest.svg)](https://github.com/deepqmc/deepqmc/releases)
[![last commit](https://img.shields.io/github/last-commit/deepqmc/deepqmc.svg)](https://github.com/deepqmc/deepqmc/commits/master)
[![license](https://img.shields.io/github/license/deepqmc/deepqmc.svg)](https://github.com/deepqmc/deepqmc/blob/master/LICENSE)
[![code style](https://img.shields.io/badge/code%20style-black-202020.svg)](https://github.com/ambv/black)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![doi](https://img.shields.io/badge/doi-10.5281%2Fzenodo.3960826-blue)](http://doi.org/10.5281/zenodo.3960826)

DeepQMC is an open-source sowftware suite for variational optimization of deep-learning molecular wave functions. It implements the simulation of electronic ground and excited states using deep neural network trial wave functions. The package is based on [JAX](https://github.com/google/jax) and [Haiku](https://github.com/deepmind/dm-haiku) and is configured through a cli build of [Hydra](https://hydra.cc/).

The program solves the molecular Hamiltonian, allowing the use of effective core potentials. Excited states are obtained via a penalty-based excited-state optimization approach. A spin penalty allows states in a fixed spin sector to be targeted.

The software suite includes a general neural network wave function ansatz, that can be configured to obtain a wide range of molecular neural network wave functions. Config files for the instantiation of variants of [PsiFormer](https://arxiv.org/abs/2211.13672), [PauliNet](https://doi.org/10.1038/s41557-020-0544-y), [FermiNet](https://link.aps.org/doi/10.1103/PhysRevResearch.2.033429) and [DeepErwin](https://arxiv.org/abs/2205.09438) can be found under `src/deepqmc/conf/ansatz`.


### Installation

Install and update to the latest release using [Pip](https://pip.pypa.io/en/stable/quickstart/):

```
pip install -U deepqmc
```

To install DeepQMC from a local Git repository run:

```
git clone https://github.com/deepqmc/deepqmc
cd deepqmc
pip install -e .[dev]
```

If Pip complains about `setup.py` not being found, please update to the latest Pip version.

The above installation will result in the CPU version of JAX. However, running DeepQMC on the GPU is highly recommended. To enable GPU support make sure to upgrade JAX to match the CUDA and cuDNN versions of your system. For most users this can be achieved with:

```
# CUDA 12 installation
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

If issues arise during the JAX installation visit the [JAX Install Guide](https://github.com/google/jax#installation).

To validate the correct installation of DeepQMC and its dependencies run:

```
deepqmc
```

### Documentation and exemplary usage

For further information about the DeepQMC package and tutorials covering the basic usage visit the [documentation](https://deepqmc.github.io).

An introduction to the methodology, implementation details and exemplary experiments can be found in the associated [software paper](https://doi.org/10.1063/5.0157512).

The penalty-based excited states approach and its implementation in DeepQMC is discussed in our recent paper on [excited state optimization](https://doi.org/10.1021/acs.jctc.4c00678).

### Citation

If you use DeepQMC for your work, please cite our implementation paper:

```
@article{10.1063/5.0157512,
    author = {Schätzle, Z. and Szabó, P. B. and Mezera, M. and Hermann, J. and Noé, F.},
    title = "{DeepQMC: An open-source software suite for variational optimization of deep-learning molecular wave functions}",
    journal = {The Journal of Chemical Physics},
    volume = {159},
    number = {9},
    pages = {094108},
    year = {2023},
    month = {09},
    issn = {0021-9606},
    doi = {10.1063/5.0157512},
    url = {https://doi.org/10.1063/5.0157512},
}

```

Experiments including excited state optimization should cite out excited state paper:

```
@article{10.1021/acs.jctc.4c00678,
    author = {Szabó, P. B. and Schätzle, Z. and Entwistle, M. and Noé, F.},
    title = "{An Improved Penalty-Based Excited-State Variational Monte Carlo Approach with Deep-Learning Ansatzes}",
    journal = {Journal of Chemical Theory and Computation},
    year = {2024},
    month = {08},
    issn = {1549-9618},
    doi = {10.1021/acs.jctc.4c00678},
    url = {https://doi.org/10.1021/acs.jctc.4c00678},
}

```

The repository can be cited as:

```
@software{deepqmc,
	author = {Hermann, J. and Schätzle, Z. and Szabó, P. B. and Mezera, M and {DeepQMC Contributors}},
	title = "{DeepQMC}",
	year = {2024},
	publisher = {Zenodo},
	copyright = {MIT},
	url = {https://github.com/deepqmc/deepqmc},
	doi = {10.5281/zenodo.3960826},
}
```
