.. _cli:

Command-line interface
======================

This section describes the command-line interface (CLI) to DeepQMC.

.. code-block:: none

    Usage: deepqmc [OPTIONS] COMMAND [ARGS]...

      DeepQMC runs quantum Monte Carlo with deep neural networks.

    Options:
      -v, --verbose  Increase verbosity.
      -q, --quiet    Suppres all output.
      --help         Show this message and exit.

    Commands:
      defaults  Print all hyperparameters and their default values.
      evaluate  Estimate total energy of an ansatz via Monte Carlo sampling.
      train     Train an ansatz with variational quantum Monte Carlo.

The CLI mirrors the package :ref:`api` in a 1-to-1 correspondence, and as such the two main entry points are the ``train`` and ``evaluate`` commands, which are just thin wrappers around the :func:`~deepqmc.train` and :func:`~deepqmc.evaluate` functions.

The usage of all the commands can be obtained with::

    deepqmc COMMAND --help

Example
-------

Create a working directory with the ``param.toml`` file, which specifies all calculation hyperparameters, see :ref:`below <hyperparameters>` for details::

    $ cat lih/param.toml
    system = 'LiH'
    ansatz = 'paulinet'

Train the wave function ansatz::

    $ deepqmc train lih --save-every 20
    converged SCF energy = -7.9846409186467
    equilibrating: 49it [00:07,  6.62it/s]
    training:   0%|  | 46/10000 [01:37<5:50:59,  2.12s/it, E=-8.0371(24)]

This creates several files in the working directory, including:

- ``events.out.tfevents.*``: Tensorboard event file
- ``fit.h5``: HDF5 file with the training trajectory
- ``chkpts/state-*.pt``: Checkpoint files with the saved state of the ansatz at particular steps

Create a file ``state.pt`` in the working directory, with the state of the ansatz that should be evaluated, for example by creating a symbolic link::

    $ ln -s chkpts/state-00040.pt lih/state.pt

Evaluate the ansatz to get an estimate of the total energy::

    $ deepqmc evaluate lih
    evaluating:  24%|▋  | 136/565 [01:12<03:40,  1.65it/s, E=-8.0396(17)]

Similar to the ``train`` command, this creates a ``sample.h5`` file with the sampled local energies in the working directory.

.. _hyperparameters:

Hyperparameters
---------------

The CLI supports specification of hyperparameters via the ``param.toml`` file, placed in ``WORKDIR``, which is passed to the individual commands. The structure of this file is derived from the :ref:`api`. All hyperparameters and their default values can be printed in the TOML format with the ``defaults`` command. For convenience, they are reproduced here:

.. literalinclude:: defaults.toml
   :language: toml
