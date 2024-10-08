.. _cli:

Command-line interface
======================

This section describes the `hydra <https://hydra.cc/>`_ based command-line interface (CLI) to DeepQMC. The tutorial exemplifies a basic training and evaluation through the command line. For more advanced functionality such as multiruns or interaction with slurm see the `hydra docs <https://hydra.cc/docs/intro/>`_.

The CLI provides simple access to the functionalities of the :class:`deepqmc` package. The main tasks comprise ``train``, ``restart`` and ``evaluate``, which are thin wrappers around the :ref:`train<api:Training and evaluation>` function.

	Available tasks:

	- ``train``:      Trains the ansatz with variational Monte Carlo.
	- ``evaluate``:   Evaluates observables (i.e. the energy) of an ansatz via Monte Carlo sampling.
	- ``restart``:    Restarts/continues the training from a stored training checkpoint.

The train function creates a directory which contains the logs as well as the hyperparameters for the training (``.hydra``). For ``restart`` and ``evaluate`` the restdir of the former training run has to be provided. Specifying arguments when executing the command will overwrite the configuration stored in the restdir. This enables changing certain parameters, such as the number of training / evaluation steps, but can result in errors if the requested hyperparameters conflict with the recovered train state.

Basics
------


A training can be run via::

    $ deepqmc hydra.run.dir=workdir


This creates several files in the working directory, including:

- ``deepqmc.log`` - Stores the console log of the run
- ``training/events.out.tfevents.*`` - Tensorboard event file
- ``training/result.h5`` - HDF5 file with the training trajectory
- ``training/state-*.pt`` - Checkpoint files with the saved state of the ansatz, optimizer and sampler at particular steps
- ``training/.hydra`` - Folder containing the `hydra <https://hydra.cc/>`_ config of the run
- ``training/pyscf_chkpts`` - Folder containing the `PySCF <https://pyscf.org/>`_ checkpoints for pretraining

The training can be continued or recoverd from a training checkpoint::

    $ deepqmc task=restart task.restdir=workdir/training

The evaluation of the energy of a trained wavefunction ansatz is obtained via::

    $ deepqmc task=evaluate task.restdir=workdir/training

This again generates a Tensorboard event file ``evaluation/events.out.tfevents.*`` and an HDF5 file ``evaluation/result.h5`` file holding the sampled local energies and other observables (see :ref:`Tutorial/Logging <logging>`)

Execution on multiple GPUs
--------------------------

DeepQMC can utilize multiple GPUs of a single host for increased performance. The algorithm is parallelised over the electron position samples, therefore the number of such samples in a batch (``electron_batch_size``) must be divisible with the number of utilized GPUs. DeepQMC relies on JAX to automatically detect and use all available GPUs, without any configuration from the user. It respects the ``CUDA_VISIBLE_DEVICES`` environment variable if it's defined, and only uses the GPUs specified there. A short log message at the beginning of the run informs the user of the number of utilized GPUs.

.. _hyperparameters:

Hyperparameters
---------------

In the following the most relevant settings for running experiments with DeepQMC are discussed.

Task
____

DeepQMC provides the above mentioned configurations for the ``train``, ``evaluate`` and ``restart`` task. In order to override default hyperparameters of the experimental setup, such as the ``sample_size`` or the number of training ``steps`` or ``pretrain_steps``, hydra provides a simple syntax::

        $ deepqmc task=train task.electron_batch_size=2048 task.steps=50000 task.pretrain_steps=5000

The working directory for logging and checkpointing is is defined through::

        $ deepqmc hydra.run.dir=workdir

Note that the working directory of an ``evaluate`` and ``restart`` task cannot match the value of their ``restdir`` option (which specifies the working directory of the previous job that we want to evaluate or restart).

Hamiltonian
___________

DeepQMC aims at solving the molecular Hamiltonian. Molecules can be selected from a range of predefined configurations located in ``.../deepqmc/src/deepqmc/conf/hamil/mol``::

        $ deepqmc hamil/mol=LiH

The predefined configurations can be extended with custom molecules. Alternatively, simple molecules can be specified on the command line::

        $ deepqmc hamil.mol.coords=[[0,0,0],[0.742,0,0]] hamil.mol.charges=[1,1] hamil.mol.charge=0 hamil.mol.spin=0 hamil.mol.unit=angstrom

To work with larger molecules, one can create custom YAML files (for examples check the ``.../deepqmc/src/deepqmc/conf/hamil/mol`` folder) and load them with::

        $ deepqmc hamil/mol=from_file hamil.mol.file=relative/path/to/molecule/file.yaml

DeepQMC implements the option to use pseudopotentials, which can be used via::

        $ deepqmc hamil.mol.coords=[[0,0,0]] hamil.mol.charges=[21] hamil.mol.charge=0 hamil.mol.spin=1 hamil.mol.unit=angstrom +hamil.ecp_type='ccECP'

Excited States
______________

DeepQMC implements penalty-based optimisation of electronic excited states. To simulate the two lowest lying states of a molecule use::

        $ deepqmc task.electronic_states=2

When simulating excited states it is recommended to pretrain with respect to orthogonal (excited) states. This is achieved by specifying a suitable cas space::

        $ deepqmc task.electronic_states=2 task.pretrain_kwargs.scf_kwargs.cas=[2,2]

To target states of a particular spin sector, a spin penalty can be applied::

        $ deepqmc task.electronic_states=2 +task.fit_fn.loss_function_factory.spin_penalty=10

Setting the spin penalty penalises high spin states, i.e. favours singlet (doublet) states over triplet (quartet) states, etc. When simulating states with higher total spin, the spin penalty is combined with setting the magnetic quantum number. For more details on the configuration of excited state calculations see [Szabo24]_. Note that when applying cas pretraining and using the spin penalty it is required to fix the spin in the calculation of the baseline to provide sensible pretraining targets.

Sampling
________

Different sampler configurations can be found in ``.../deepqmc/src/deepqmc/conf/task/sampler_factory``. A typical usecase would be to pick as sampler form these configurations and, if required, change some argument from the command line::

        $  deepqmc task/sampler_factory=decorr_langevin task.sampler_factory.elec_sampler.samplers.0.length=30

Optimization
____________

For the optimization either `KFAC <https://kfac-jax.readthedocs.io/en/latest/>`_ or optimizers from `optax <https://optax.readthedocs.io/en/latest/>`_ may be used. While the use of `KFAC <https://kfac-jax.readthedocs.io/en/latest/>`_ is highly recommended due to the significantly improved convergence, at times it can be useful to run with other optimizers such as `AdamW <https://optax.readthedocs.io/en/latest/api.html#adamw>`_::

        $ deepqmc task/opt=adamw

Ansatz
______

The hyperparameters of the training and the wave function ansatz are specified through hydra config files. Predefined ansatzes can be found in ``.../deepqmc/src/deepqmc/conf/ansatz`` and selected via::

    $ deepqmc ansatz=psiformer

The hyperparameters of such a predefined ansatz can also be overwritten at the command line::

    $ deepqmc ansatz=psiformer ansatz.omni_factory.gnn_factory.n_interactions=2

For convenience the configuration of the ``default`` ansatz is reproduced here:

.. literalinclude:: default.yaml
   :language: yaml
