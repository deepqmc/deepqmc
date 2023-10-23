# DeepQMC

![checks](https://img.shields.io/github/actions/workflow/status/deepqmc/deepqmc/tests.yaml?label=tests)
[![coverage](https://img.shields.io/codecov/c/github/deepqmc/deepqmc.svg)](https://codecov.io/gh/deepqmc/deepqmc)
![python](https://img.shields.io/pypi/pyversions/deepqmc.svg)
[![pypi](https://img.shields.io/pypi/v/deepqmc.svg)](https://pypi.org/project/deepqmc/)
[![commits since](https://img.shields.io/github/commits-since/deepqmc/deepqmc/latest.svg)](https://github.com/deepqmc/deepqmc/releases)
[![last commit](https://img.shields.io/github/last-commit/deepqmc/deepqmc.svg)](https://github.com/deepqmc/deepqmc/commits/master)
[![license](https://img.shields.io/github/license/deepqmc/deepqmc.svg)](https://github.com/deepqmc/deepqmc/blob/master/LICENSE)
[![code style](https://img.shields.io/badge/code%20style-black-202020.svg)](https://github.com/ambv/black)
[![doi](https://img.shields.io/badge/doi-10.5281%2Fzenodo.3960826-blue)](http://doi.org/10.5281/zenodo.3960826)

DeepQMC implements variational quantum Monte Carlo for electrons in molecules, using deep neural networks as trial wave functions. The package is based on [JAX](https://github.com/google/jax) and [Haiku](https://github.com/deepmind/dm-haiku). Besides the core functionality, it contains an implementation of a flexible neural network wave function ansatz, that can be configured to obtain a broad range of molecular neural network wave functions. Config files for the instantiation of variants of [PauliNet](https://doi.org/10.1038/s41557-020-0544-y), [FermiNet](https://link.aps.org/doi/10.1103/PhysRevResearch.2.033429), [DeepErwin](https://arxiv.org/abs/2205.09438) and [PsiFormer](https://arxiv.org/abs/2211.13672) can be found under `src/deepqmc/conf/ansatz`.

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

# CUDA 11 installation
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

If issues arise during the JAX installation visit the [JAX Install Guide](https://github.com/google/jax#installation).

### Documentation and exemplary usage

For further information about the DeepQMC package and tutorials covering the basic usage visit the [documentation](https://deepqmc.github.io).

An introduction to the methodology and exemplary experiments can be found in the associated [software paper](https://doi.org/10.1063/5.0157512).


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
The repository can be cited as:

```
@software{deepqmc,
	author = {Hermann, J. and Schätzle, Z. and Szabó, P. B. and Mezera, M and {DeepQMC Contributors}},
	title = "{DeepQMC}",
	year = {2023},
	publisher = {Zenodo},
	copyright = {MIT},
	url = {https://github.com/deepqmc/deepqmc},
	doi = {10.5281/zenodo.7503172},
}
```

