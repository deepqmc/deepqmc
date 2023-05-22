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

DeepQMC implements variational quantum Monte Carlo for electrons in molecules, using deep neural networks as trial wave functions. The package is based on [JAX](https://github.com/google/jax) and [Haiku](https://github.com/deepmind/dm-haiku). Besides the core functionality, it contains an implementation of a flexible neural network wave function ansatz, that can be configured to obtain a broad range of molecular neural network wave functions. Config files for the instantiation of variants of [PauliNet](https://doi.org/10.1038/s41557-020-0544-y), [FermiNet](https://link.aps.org/doi/10.1103/PhysRevResearch.2.033429) and [DeepErwin](https://arxiv.org/abs/2205.09438) can be found under `src/deepqmc/conf/ansatz`.

### Installation

Install and update to the latest release using [Pip](https://pip.pypa.io/en/stable/quickstart/):

```
pip install -U deepqmc -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

To install DeepQMC from a local Git repository run:

```
git clone https://github.com/deepqmc/deepqmc
cd deepqmc
pip install -e .[dev] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

If pip complains about `setup.py` not being found, please make sure to update to the latest pip version.

### Documentation and exemplary usage

For further information about the DeepQMC package and tutorials covering the basic usage visit the [documentation](https://deepqmc.github.io).

### Citation

This repository can be cited as:
```
@software{deepqmc,
	author = {Jan Hermann and
		  Zeno Schätzle and
		  Peter Bernát Szabó and
		  Matěj Mezera and
		  {DeepQMC Contributers}},
	title = {{DeepQMC}},
	year = {2023},
	publisher = {Zenodo},
	copyright = {MIT},
	url = {https://github.com/deepqmc/deepqmc},
	doi = {10.5281/zenodo.7503172},
}
```

