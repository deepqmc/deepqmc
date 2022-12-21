.. _cli:

Command-line interface
======================

This section describes the `hydra <https://hydra.cc/>`_ based command-line interface (CLI) to DeepQMC. The tutorial exemplifies a basic training and evaluation through the command line. For more advanced functionality such as multiruns or interaction with slurm see the `hydra docs <https://hydra.cc/docs/intro/>`_.

The CLI provides simple access to the functionalities of the :class:`deepqmc` package. The main tasks comprise ``train``, ``restart`` and ``evaluate``, which are thin wrappers around the :ref:`train<api:Training and evaluation>` function.

	Available tasks:

	- ``train``:      Trains the ansatz with variational Monte Carlo.
	- ``evaluate``:   Evaluates the total energy of an ansatz via Monte Carlo sampling.
	- ``restart``:    Restarts/continues the training from a stored training checkpoint.
	- ``defaults``:   Collects the default hyperparameters for the main DeepQMC functions.

The train function creates a directory which contains the logs as well as the hyperparameters for the training (``.hydra``). For ``restart`` and ``evaluate`` the restdir of the former training run has to be provided. Specifying arguments when executing the command will overwrite the configuration stored in the restdir. This enables changing certain parameters, such as the number of training / evaluation steps, but can result in errors if the requested hyperparameters conflict with the recovered train state. 

Example
-------


A training can be run via::

    $ deepqmc hydra.run.dir=workdir hamil.mol.name=LiH


This creates several files in the working directory, including:

- ``training/events.out.tfevents.*`` - Tensorboard event file
- ``training/result.h5`` - HDF5 file with the training trajectory
- ``training/state-*.pt`` - Checkpoint files with the saved state of the ansatz at particular steps

The training can be continued or recoverd from a training checkpoint::

    $ deepqmc task=restart task.restdir=workdir

The evaluation of the energy of a trained wavefunction ansatz is obtained via::

    $ deepqmc task=evaluate task.restdir=workdir

This again generates a Tensorboard event file ``evaluation\events.out.tfevents.*`` and an HDF5 file ``evaluation\result.h5`` file holding the sampled local energies.

.. _hyperparameters:

Hyperparameters
---------------

A list of the most relevant hyperparameters and their default values can be obtained with the ``task=defaults`` argument. The defaults are automatically colleceted from the main deepqmc components and logged in YAML format. For convenience, they are reproduced here:

.. literalinclude:: defaults.yaml
   :language: yaml
