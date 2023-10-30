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
- `Haiku <https://github.com/deepmind/dm-haiku>`_ is a neural network library for JAX that enables object oriented programming of models in the purely functional jax environment.
- `uncertainties <http://uncertainties-python-package.readthedocs.io>`_ helps with propagation of uncertainties in calculations.
- `TQDM <https://github.com/tqdm/tqdm>`_ provides progress bars.
- `kfac <https://github.com/deepmind/kfac-jax>`_ is a jax implementation of the KFAC second order optimizer.
- `optax <https://github.com/deepmind/optax>`_ provides optimizers and loss functions for jax.
- `h5py <https://www.h5py.org>`_ handles IO for `HDF5 <http://hdfgroup.org>`_ files.
- `Tensorboard <https://www.tensorflow.org/tensorboard>`_ is a practical tool for monitoring training of neural networks.
- `hydra <https://hydra.cc/>`_ helps with constructing command-line interfaces.
- `PySCF <http://pyscf.org>`_ implements quantum chemistry methods in Python. This is used to obtain the baseline for pretraining.

Optional dependencies
~~~~~~~~~~~~~~~~~~~~~

The following dependencies are used for development and their installation must be requested explicitly, as documented below.

- `black <https://github.com/psf/black>`_ formats code according to PEP 8 standard.
- `flake8 <https://github.com/PyCQA/flake8>`_ implement style guidelines.
- `isort <https://github.com/PyCQA/isort>`_ helps to keep consistent order of imports.
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

In order to install DeepQMC, an appropriate version of the JAX library has to be installed first.
The necessary version of JAX depends on whether one intends to run DeepQMC on GPUs or CPUs, and in the case of GPUs the version of the CUDA libraries installed on the machine.

Running on GPUs (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Running DeepQMC on GPUs is highly recommended as, similar to most other ML applications, this results in far superior performance.
To enable GPU support, make sure to install the appropratie, CUDA enabled JAX version that matches the CUDA and cuDNN versions on your machine.
For most users this can be achieved with one of the following commands::

    $ # CUDA 12 installation
    $ pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    $ # CUDA 11 installation
    $ pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

To determine the CUDA version installed on your machine check e.g. the output of the command ``nvidia-smi``.
If issues arise during the installation of JAX, or if errors related to CUDA or cuDNN occur at runtime, please visit the `JAX Install Guide <https://github.com/google/jax#installation>`_.

Running on CPUs
~~~~~~~~~~~~~~~

To install a CPU-only version of JAX, use the command::

    $ pip install --upgrade "jax[cpu]"

Installing the DeepQMC package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the appropriate version of JAX is installed, the latest official release of DeepQMC can be installed from the Python Package Index using `Pip <https://pip.pypa.io/en/stable/quickstart/>`_::

    $ pip install --upgrade deepqmc

Developing
----------

In order to have access to the source code and stay up-to-date with the latest developments, DeepQMC can be installed directly from the https://github.com/deepqmc/deepqmc GitHub repository.

To install DeepQMC from the Git repository run::

    $ git clone https://github.com/deepqmc/deepqmc
    $ cd deepqmc
    $ pip install -e .[dev]

Note that the ``-e`` option installs the repository in editable mode and the ``.[dev]`` specification includes the optional dependencies for development.

If `Pip <https://pip.pypa.io/en/stable/quickstart/>`_ complains about ``setup.py`` not being found, please update pip to the latest version.

In order to contribute directly to the repository, the pull requests and code have to conform to our `contributing guidelines <https://github.com/deepqmc/deepqmc/blob/master/CONTRIBUTING.md>`_.
Most of these can be automatically checked/enforced using our `pre-commit hooks <https://pre-commit.com/>`_, which can be enabled by issuing the following command from the root directory of the repository::

    $ pre-commit install
