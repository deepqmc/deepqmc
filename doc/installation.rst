.. _installation:

Installation
============

Python version
--------------

DeepQMC is compatible with Python 3.6, but Python 3.7 is recommended.

Dependencies
------------

These packages will be installed automatically when installing DeepQMC.

- `NumPy <https://numpy.org>`_ is an essential library for numerical computation in Python.
- `PyTorch <https://pytorch.org>`_ is one of the most popular Python frameworks for deep learning.
- `TOML <https://github.com/uiri/toml>`_ implements the `TOML <https://en.wikipedia.org/wiki/TOML>`_ configuration file format in Python.

Optional dependencies
~~~~~~~~~~~~~~~~~~~~~

The following dependencies are required only by non-essential parts of DeepQMC and their installation must be requested explicitly, as documented below.

- `SciPy <https://www.scipy.org>`_ provides essential numerical algorithms.
- `PySCF <http://pyscf.org>`_ implements quantum chemistry methods in Python. This is used to obtain the HF orbitals.
- `Tensorboard <https://www.tensorflow.org/tensorboard>`_ is a practical tool for monitoring training of neural networks.
- `TQDM <https://github.com/tqdm/tqdm>`_ provides progress bars.

Virtual environments
--------------------

It is a good practice to separate the dependencies of different Python projects with the use of virtual environments::

   $ mkdir myproject
   $ cd myproject
   $ python3 -m venv venv
   $ source venv/bin/activate

This code creates a virtual environment in the `venv` directory and actives it. As a result, the `python` and `pip` executables available in the shell are now in the `venv/bin` directory, and any package and its dependencies are installed locally.

Install DeepQMC
---------------

Within the activated virtual environment, DeepQMC can be installed with::

    $ pip install deepqmc

To install all optional dependencies, use::

    $ pip install deepqmc[wf,train]

Developing DeepQMC
------------------

To install DeepQMC from a local Git repository, use `Poetry <https://python-poetry.org>`_::

    $ git clone https://github.com/noegroup/deepqmc
    $ cd deepqmc
    $ poetry install -E all

In addition to all the optional dependencies above, this also installs `pytest`, `coverage`, and `click`.
