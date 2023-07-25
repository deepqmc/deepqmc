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
- `h5py <https://www.h5py.org>`_ handles IO for `HDF5 <http://hdfgroup.org>`_ files.
- `Tensorboard <https://www.tensorflow.org/tensorboard>`_ is a practical tool for monitoring training of neural networks.
- `hydra <https://hydra.cc/>`_ helps with constructing command-line interfaces.
- `PySCF <http://pyscf.org>`_ implements quantum chemistry methods in Python. This is used to obtain the baseline for pretraining.

Optional dependencies
~~~~~~~~~~~~~~~~~~~~~

The following dependencies are used for development and their installation must be requested explicitly, as documented below.

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

The latest official release of DeepQMC can be installed from the Python Package Index using `Pip <https://pip.pypa.io/en/stable/quickstart/>`_::

    $ pip install -U deepqmc

Developing
----------

In order to have access to the source code and stay up-to-date with the latest developments, DeepQMC can be installed directly from the https://github.com/deepqmc/deepqmc GitHub repository. 

To install DeepQMC from the Git repository run::

    $ git clone https://github.com/deepqmc/deepqmc
    $ cd deepqmc
    $ pip install -e .[dev] 

Note that the ``-e`` option installs the repository in editable mode and the ``.[dev]`` specification includes the optional dependencies for development.

If `Pip <https://pip.pypa.io/en/stable/quickstart/>`_ complains about ``setup.py`` not being found, please update pip to the latest version.

The above installation will result in the CPU version of JAX. However, running DeepQMC on the GPU is highly recommended. To enable GPU support make sure to upgrade JAX to match the CUDA and cuDNN versions of your system. For most users this can be achieved with::

    $ # CUDA 12 installation
    $ pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    $ # CUDA 11 installation
    $ pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

If issues arise during the installation of JAX, or if errors related to CUDA or cuDNN occur at runtime, please visit the `JAX Install Guide <https://github.com/google/jax#installation>`_.
