.. _installation:

:tocdepth: 2

Installation
============

Python version
--------------

DeepQMC is compatible with Python 3.9 and higher.

Dependencies
------------


These packages will be installed automatically when installing DeepQMC.

- `JAX <https://github.com/google/jax>`_ is a popular Python framework that combines NumPy, Autograd and XLA to give highly efficient just-in-time compiled code with GPU and TPU support.
- `Haiku <https://github.com/deepmind/dm-haiku>`_ is a neural network libary for JAX that enables object oriented programming of models in the purely functional jax environment.
- `uncertainties <http://uncertainties-python-package.readthedocs.io>`_ helps with propagation of uncertainties in calculations.
- `TQDM <https://github.com/tqdm/tqdm>`_ provides progress bars.
- `kfac <https://github.com/deepmind/kfac-jax>`_ is a jax implementation of the KFAC second order optimizer.
- `optax <https://github.com/deepmind/optax>`_ provides optimizers and loss functions for jax.
- `e3nn-jax <https://github.com/e3nn/e3nn-jax>`_ provides functionality for E(3) equivariant neural networks.
- `h5py <https://www.h5py.org>`_ handles IO for `HDF5 <http://hdfgroup.org>`_ files.
- `Tensorboard <https://www.tensorflow.org/tensorboard>`_ is a practical tool for monitoring training of neural networks.
- `hydra <https://hydra.cc/>`_ helps with constructing command-line interfaces.
- `PySCF <http://pyscf.org>`_ implements quantum chemistry methods in Python. This is used to obtain the baseline for pretraining.

Optional dependencies
~~~~~~~~~~~~~~~~~~~~~

The following dependencies are used for developement and their installation must be requested explicitly, as documented below.

- `black <https://github.com/psf/black>`_ formats code according to PEP 8 standard.
- `flake8 <https://github.com/PyCQA/flake8>`_ implement style guidelines.
- `isort <https://github.com/PyCQA/isort>`_ helps to keep consitent order of imports.
- `pytest <https://docs.pytest.org/en/7.2.x>`_ for testing the code.
- `pytest-regressions <https://github.com/ESSS/pytest-regressions>`_ for testing numerical regressions in the code.
- `pydocstyle <https://github.com/PyCQA/pydocstyle>`_  check compliance with Python docstring conventions.
- `Coverage.py <https://github.com/nedbat/coveragepy>`_  measures code coverage for testig.

Virtual environments
--------------------

It is a good practice to separate the dependencies of different Python projects with the use of virtual environments::

   $ mkdir myproject
   $ cd myproject
   $ python3 -m venv venv
   $ source venv/bin/activate

This code creates a virtual environment in the ``venv`` directory and actives it. As a result, the ``python`` and ``pip`` executables available in the shell are now in the ``venv/bin`` directory, and any package and its dependencies are installed locally.

Install DeepQMC
---------------

Within the activated virtual environment, DeepQMC can be installed with::

    $ pip install -U deepqmc -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Where specifying the additional URL is necessary, as the CUDA compatible versions of `jaxlib <https://github.com/google/jax>`_ are not hosted `pypi <https://pypi.org/>`_.
To install all optional dependencies, use::

    $ pip install -U deepqmc[dev] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Developing
----------

To install DeepQMC from a local Git repository run::

    $ git clone https://github.com/deepqmc/deepqmc
    $ cd deepqmc
    $ pip install -e .[dev] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
