.. _cli:

Command-line interface
======================

This section describes the `hydra <https://hydra.cc/>`_ based command-line interface (CLI) to DeepQMC. The tutorial exemplifies a basic training and evaluation through the command line. For more advanced functionality such as multiruns or interaction with slurm see the `hydra docs <https://hydra.cc/docs/intro/>`_.

The CLI provides simple access to the functionalities of the :class:`deepqmc` package. The main tasks comprise ``train``, ``restart`` and ``evaluate``, which are thin wrappers around the :ref:`train<api:Training and evaluation>` function.

	Available tasks:

	- ``train``:      Trains the ansatz with variational Monte Carlo.
	- ``evaluate``:   Evaluates the total energy of an ansatz via Monte Carlo sampling.
	- ``restart``:    Restarts/continues the training from a stored training checkpoint.

The train function creates a directory which contains the logs as well as the hyperparameters for the training (``.hydra``). For ``restart`` and ``evaluate`` the restdir of the former training run has to be provided. Specifying arguments when executing the command will overwrite the configuration stored in the restdir. This enables changing certain parameters, such as the number of training / evaluation steps, but can result in errors if the requested hyperparameters conflict with the recovered train state. 

Example
-------


A training can be run via::

    $ deepqmc hydra.run.dir=workdir


This creates several files in the working directory, including:

- ``training/events.out.tfevents.*`` - Tensorboard event file
- ``training/result.h5`` - HDF5 file with the training trajectory
- ``training/state-*.pt`` - Checkpoint files with the saved state of the ansatz at particular steps

The training can be continued or recoverd from a training checkpoint::

    $ deepqmc task=restart task.restdir=workdir

The evaluation of the energy of a trained wavefunction ansatz is obtained via::

    $ deepqmc task=evaluate task.restdir=workdir

This again generates a Tensorboard event file ``evaluation/events.out.tfevents.*`` and an HDF5 file ``evaluation/result.h5`` file holding the sampled local energies.

.. _hyperparameters:

Hyperparameters
---------------

In the following the most relevant settings for running experiments with DeepQMC are discussed.

Task
____

DeepQMC provides the above mentioned conigurations for the ``train``, ``evaluate`` and ``restart`` task. In order to override default hyperparameters of the experimental setup, such as the ``sample_size`` or the number of training ``steps`` or ``pretrain_steps``, hydra provides a simple syntax::

        $ deepqmc task=train task.sample_size=2048 task.steps=50_000 +task.pretrain_steps=1000

The working directory for logging and checkpointing is is defined through::

        $ deepqmc hydra.run.dir=workdir

Note that the working directory of an ``evaluate`` and ``restart`` task cannot match the value of their ``restdir`` option (which specifies the working directory of the previous job that we want to evaluate or restart).

Hamiltonian
___________

DeepQMC aims at solving the molecular Hamiltonian. Molecules can be selected from a range of predefined configurations located in ``.../deepqmc/src/deepqmc/conf/hamil/mol``::

        $ deepqmc hamil/mol=LiH

The predefined configurations can be extended with custom molecules. Alternatively molecules can be specified on the command line::

        $ deepqmc hamil.mol.coords=[[0,0,0],[0.742,0,0]] hamil.mol.charges=[1,1] hamil.mol.charge=0 hamil.mol.spin=0 hamil.mol.unit=angstrom

Furthermore, DeepQMC implements the option to use pseudopotentials, which can be used via::

        $ deepqmc hamil.mol.coords=[[0,0,0]] hamil.mol.charges=[21] hamil.mol.charge=0 hamil.mol.spin=1 hamil.mol.unit=angstrom +hamil.mol.pp_type='ccECP'

Sampling
________

Different sampler configurations can be found in ``.../deepqmc/src/deepqmc/conf/task/sampler``. A typical usecase would be to pick as sampler form these configurations and, if required, change some argument from the command line::

        $deepqmc task/sampler=decorr_langevin task.sampler.0.length=30

Optimization
____________

For the optimization either `KFAC <https://kfac-jax.readthedocs.io/en/latest/>`_ or optimizers from `optax <https://optax.readthedocs.io/en/latest/>`_ may be used. While the use of `KFAC <https://kfac-jax.readthedocs.io/en/latest/>`_ is highly recommended due to the significantly imporved convergence, at times it can be useful to run with other optimizers such as `AdamW <https://optax.readthedocs.io/en/latest/api.html#adamw>`_::

        $ deepqmc task/opt=adamw

Ansatz
______

The hyperparameters of the training and the wave function ansatz are specified through hydra config files. Predefined ansatzes can be found in ``.../deepqmc/src/deepqmc/conf/ansatz`` and selected via::

    $ deepqmc ansatz=ferminet

The hyperparameters of such a predifined ansatz can also be overwritten at the command line::

    $ deepqmc ansatz=ferminet ansatz.omni_factory.gnn_factory.n_interactions=2

For convenience the configuration of the ``default`` ansatz is reproduced here:

.. literalinclude:: default.yaml
   :language: yaml
