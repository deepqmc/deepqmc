.. _cli:

Command-line interface
======================

The most convenient and reproducible way of running an experiment with DeepQMC is through the `hydra <https://hydra.cc/>`_ based command-line interface (CLI).
The tutorial exemplifies a basic training and evaluation through the command line.
For more advanced functionality such as multiruns or interaction with slurm see the `hydra docs <https://hydra.cc/docs/intro/>`_.

The CLI provides simple access to the functionalities of the DeepQMC package. The main tasks comprise ``train``, ``restart`` and ``evaluate``, which are thin wrappers around the :ref:`train<api:Training and evaluation>` function.

	Available tasks:

	- ``train``:      Trains the ansatz with variational Monte Carlo.
	- ``evaluate``:   Evaluates observables (i.e. the energy) of an ansatz via Monte Carlo sampling.
	- ``restart``:    Restarts/continues the training from a stored training checkpoint.

The train function creates a directory which contains the logs as well as the hyperparameters for the training (``.hydra``). For ``restart`` and ``evaluate`` the restdir of the former training run has to be provided. Specifying arguments when executing the command will overwrite the configuration stored in the restdir. This enables changing certain parameters, such as the number of training / evaluation steps, but can result in errors if the requested hyperparameters conflict with the recovered train state.

Basics
------


A training can be run via::

    $ deepqmc hydra.run.dir=workdir


Running the plain deepqmc command launches a default job on the LiH molecule, which is good for testing the functionality of the code.
The `hydra.run.dir=workdir` arguments sets the working directory of the job to "workdir".
In the working directory several files are created, including:

- ``deepqmc.log`` - Stores the console log of the run
- ``training/events.out.tfevents.*`` - Tensorboard event file
- ``training/result.h5`` - HDF5 file with the training trajectory
- ``training/state-*.pt`` - Checkpoint files with the saved state of the ansatz, optimizer and sampler at particular steps
- ``training/.hydra`` - Folder containing the `hydra <https://hydra.cc/>`_ config of the run
- ``training/pyscf_chkpts`` - Folder containing the `PySCF <https://pyscf.org/>`_ checkpoints for pretraining

The evaluation of the energy of a trained wavefunction ansatz is obtained via::

    $ deepqmc task=evaluate task.restdir=workdir/training

The training can be continued or recoverd from a training checkpoint::

    $ deepqmc task=restart task.restdir=workdir/training

This again generates a Tensorboard event file ``evaluation/events.out.tfevents.*`` and an HDF5 file ``evaluation/result.h5`` file holding the sampled local energies and other observables (see :ref:`Tutorial/Logging <logging>`).
If a training is evaluated or continued the hydra configurations are chained together, with the latter runs referring to the configs of the former.
Note that moving the working directories during subsequent training, restarts and evaluations may lead to broken references in the config paths in ``training/.hydra`` and should be avoided (if necessary manual correction of the paths is required).


Execution on multiple GPUs
--------------------------

DeepQMC can utilize multiple GPUs increased performance.
The algorithm is parallelised over the electron position samples, therefore the number of such samples in a batch (``electron_batch_size``) must be divisible with the number of utilized GPUs.
DeepQMC relies on JAX to automatically detect and use all available GPUs, without any configuration from the user.
It respects the ``CUDA_VISIBLE_DEVICES`` environment variable if it's defined, and only uses the GPUs specified there.
A short log message at the beginning of the run informs the user of the number of utilized GPUs.

.. code-block:: text

        INFO:deepqmc.app: Running on X GPU_NAME with Y processes

.. _hyperparameters:

Hyperparameters
---------------

In the following the most relevant settings for running experiments with DeepQMC are discussed.
Various application examples with a more in depths explanation of the arguments are provided under the :ref:`examples <examples>` page.


Task
____

DeepQMC provides the above mentioned configurations for the ``train``, ``evaluate`` and ``restart`` task.
In order to override default hyperparameters of the experimental setup, such as the ``sample_size`` or the number of training ``steps`` or ``pretrain_steps``, hydra provides a simple syntax::

        $ deepqmc task=train task.electron_batch_size=2048 task.steps=50000 task.pretrain_steps=5000

The working directory for logging and checkpointing is is defined through::

        $ deepqmc hydra.run.dir=workdir

If no Logging directory is provided a directory ``outputs/YEAR-MONTH-DAY/HOUR-MINUTE-SECOND`` is being used.
Note that the working directory of an ``evaluate`` and ``restart`` task cannot match the value of their ``restdir`` option.

Hamiltonian
___________

DeepQMC aims at solving the molecular Hamiltonian. Molecules can be selected from a range of predefined configurations located in ``.../deepqmc/src/deepqmc/conf/hamil/mol``::

        $ deepqmc hamil/mol=LiH

The hydra syntax allows specifying molecules on the command line::

        $ deepqmc hamil.mol.coords=[[0,0,0],[0.742,0,0]] hamil.mol.charges=[1,1] hamil.mol.charge=0 hamil.mol.spin=0 hamil.mol.unit=angstrom

In practice it is often more convenient to create custom YAML files (for examples check the ``.../deepqmc/src/deepqmc/conf/hamil/mol`` folder) and load them with::

        $ deepqmc hamil/mol=from_file hamil.mol.file=relative/path/to/molecule/file.yaml

Pseudopotentials and pseudohamiltonains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DeepQMC implements the option to use pseudopotentials [Burkatzki07]_ [Bennett17]_ and pseudohamiltonains [Ichibha23]_ [Fu26]_.
Pseudopotentials and pseudohamiltonians are enabled via the `ecp_type` and `ecp_mask` arguments of the hamiltonian.
For the ccECP type pseudopotential use `ecp_type='ccECP'`::

        $ deepqmc hamil/mol='Sc' +hamil.ecp_type='ccECP'

For the pseudohamiltonians use `ecp_type=PH`::

        $ deepqmc hamil/mol='Fe' +hamil.ecp_type='PH'

DeepQMC supports the Gaussian-type pseudopotentials from PySCF as well and provides pseudohamiltonians for S, Cr, Mn, Fe, Co, Ni, Cu and Zn.

Furthermore, DeepQMC uses the `folx <https://github.com/microsoft/folx>`_ package to utilize the forward laplacian framework [Li24]_.
The forward Laplacian significantly accelerates the computation of the Laplacian at the cost of a an increased memory footprint, resulting in about a 2x overall speed-up of the simulation of intermediately sized systems.
The use of the forward laplacian is always recommended if compatible with the computational setup::

        $ deepqmc hamil=qc_forward_laplacian

.. _sampling:

Sampling
________

Different sampler configurations can be found in ``.../deepqmc/src/deepqmc/conf/task/sampler_factory``.
A typical usecase would be to pick as sampler form these configurations and, if required, change some argument from the command line::

        $  deepqmc task/sampler_factory=decorr_langevin task.sampler_factory.elec_sampler.samplers.0.length=30

Optimization
____________

For the optimization either `KFAC <https://kfac-jax.readthedocs.io/en/latest/>`_ or optimizers from `optax <https://optax.readthedocs.io/en/latest/>`_ may be used.
While the use of `KFAC <https://kfac-jax.readthedocs.io/en/latest/>`_ is highly recommended due to the significantly improved convergence, at times it can be useful to run with other optimizers such as `AdamW <https://optax.readthedocs.io/en/latest/api.html#adamw>`_::

        $ deepqmc task/opt=adamw

Excited States
______________

DeepQMC implements penalty-based optimisation of electronic excited states.

Excited states with via orthogonalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To simulate the two lowest lying states of a molecule use::

        $ deepqmc task.electronic_states=2

When simulating excited states it can be useful to pretrain with respect to orthogonal (excited) states. This is achieved by specifying a suitable cas space::

        $ deepqmc task.electronic_states=2 +task.pretrain_kwargs.scf_kwargs.cas=[2,2]

Targeting spin sectors with a spin penalty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To target states of a particular spin sector, a spin penalty can be applied::

        $ deepqmc task.electronic_states=2 +task.loss_function_factory.spin_penalty=10

Setting the spin penalty penalises high spin states, i.e. favours singlet (doublet) states over triplet (quartet) states, etc.
When simulating states with higher total spin, the spin penalty is combined with setting the magnetic quantum number, i.e. setting ``mol.spin=2`` in the molecule configuration for a triplet.
Note that when combined with cas pretraining it is required to fix the spin in the calculation of the baseline to provide sensible pretraining targets.

For more details on the configuration of excited state calculations see [Szabo24]_.

Sharing parameters across excited states
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wave function parameters can be shared across excited states by using the ``merge_keys`` argument.
This accepts a list of strings that are matched against the parameter names, i.e. all the parameters of the graph neural network / transformer layers::

        $ deepqmc ansatz=psiformer task.electronic_states=2 +task.merge_keys="['embedding', 'gnn']"

A list of the shared parameters will be logged at the beginning of the training.

Transferable Training
_____________________

DeepQMC implements geometrically transferable training, that is a single ansatz can be trained across multiple molecular configurations.
This significantly reduces the computational cost over equivalent independent single-point simulations, improves relative energies and can be used for interpolation of entire potential energy surfaces, including for ab initio geometry optimization [Szabo26]_.
Transferable simulation can be done in various modes and can be combined with excited state simulation and spin penalties.
Here we provide a basic introduction. For more examples see the :ref:`examples <examples>` page.

Sharing parameters for a fixed dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest version of geometric transferability can be thought of as sharing wave function parameters across multiple parallel DeepQMC simulations on different geometric configurations of the same molecule.
Therefore, we instantiate a joint training run on multiple fixed geometries by providing the optional ``mols`` argument to the train function.
The most convenient way to instantiate a transferable training run is by using the ``task=train_transferable`` config and providing a directory in which molecule configurations are stored as yaml files.
Here we generate a dataset composed of two configurations of the LiH molecule::

        $ mkdir -p mols_dir

        $ cat > mols_dir/LiH_eq.yaml <<EOF
        coords: [[0.0, 0.0, 0.0], [3.014, 0.0, 0.0]]
        charges: [3, 1]
        charge: 0
        spin: 0
        EOF

        $ cat > mols_dir/LiH_stretch.yaml <<EOF
        coords: [[0.0, 0.0, 0.0], [6, 0.0, 0.0]]
        charges: [3, 1]
        charge: 0
        spin: 0
        EOF

We then run a transferable DeepQMC simulation::

    $ deepqmc hydra.run.dir=workdir_transferable task=train_transferable  task.seed=42 ansatz=transpsiformer task.molecule_batch_size=2 hamil/mol=LiH task.mols.directory=mols_dir

Additional to providing the mols directory the molecule in the Hamiltonian needs to be set accordingly (this molecule is used for determining shapes of the wave function etc.).
Note that all the molecule configurations in ``mols_dir`` as well as the molecule specified in the Hamiltonian need to have the same charges, total charge, spin, etc. that is have to be equivalent up to their geometry.

We use the transpsiformer ansatz described in [Schaetzle25]_, which is an extension of the Psiformer [Glehn23]_ that explicitly accounts for changes in the nuclear geometry.

Setting ``molecule_batch_size=2`` means that two molecules per iteration are run in parallel, each with ``electron_batch_size`` many walkers.
For large datasets it is possible to work only on a subset of the molecular geometries on each iteration (i.e. ``molecule_batch_size<len(mols)``) and cycle through the data iteratively.
The exact choice of ``electron_batch_size`` and ``molecule_batch_size`` may depend on the experimental setup and differs among practitioners.
We typically use a ``molecule_batch_size`` of 4-32, with an inversely proportional ``electron_batch_size`` of 2048-256 scaled to optimize GPU bandwidth.

In this training mode individual markov chains are retained for each geometry.
Therefore, if the ``molecule_batch_size`` is significantly smaller than the number of molecules in the training set, walkers can go stale and it can be useful to increase the number of :ref:`decorrelation steps <sampling>`.

After a shared optimization the transferable ansatz can be evaluated on the geometries of interest independently:

    $ deepqmc hydra.run.dir=workdir_eval task=evaluate task.restdir=workdir_transferable/training task.molecule_batch_size=1 hamil/mol=from_file hamil.mol.directory=mols_dir/LiH_eq

Dynamic geometry sampling
~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of training on a fixed set of molecular geometries, DeepQMC can dynamically resample the nuclear geometry throughout the optimization, directly in internal (bond length, angle and dihedral) coordinates.
This is the approach used for learning continuous potential energy surfaces and for ab initio geometry optimization [Szabo26]_, see also :ref:`api:Nuclear geometry`.

A dynamically sampled geometry is configured via the ``nuc_sampler`` argument of the sampler factory.
A convenient way to do this is to use a :class:`~deepqmc.sampling.nuclei_samplers.ZMatrixSampler` built from a :class:`~deepqmc.geom.zmatrix.StochasticZMatrixTemplate`.
The template specifies, for every atom (in the same order as the nuclear charges of the Hamiltonian's molecule), which of its bond length, bond angle and dihedral angle relative to previously listed atoms are resampled every iteration, and from which noise distribution.
Custom sampler configurations are added as YAML files under ``.../deepqmc/src/deepqmc/conf/task/sampler_factory/nuc_sampler``.
For example, the following configuration continuously stretches and compresses the Li-H bond of the LiH molecule around its reference length, clipped so that it never becomes unphysically short (bond lengths are in bohr, the atomic unit of length)::

        $ cat > src/deepqmc/conf/task/sampler_factory/nuc_sampler/LiH_zmat.yaml <<EOF
        _target_: deepqmc.sampling.nuclei_samplers.ZMatrixSampler
        _partial_: true
        z_matrix_template:
          _target_: deepqmc.geom.zmatrix.StochasticZMatrixTemplate.from_simplified_config
          lines:
            - charge: 3
              atom_idxs: [null, null, null]
              distribution_factories: [null, null, null]
            - charge: 1
              atom_idxs: [0, null, null]
              distribution_factories:
                - _target_: deepqmc.geom.zmatrix.ClippedNormalDistributionFactory
                  scale: 2.0
                  low: 1.5
                - null
                - null
        EOF

Each entry of ``lines`` corresponds to one atom, with ``atom_idxs`` giving the (zero-based) indices of the previously listed atoms its bond, angle and dihedral are measured from (``null`` where not applicable, e.g. the first atom of a Z-matrix never has any of these, and the second, as here, has at most a bond).
:class:`~deepqmc.geom.zmatrix.ClippedNormalDistributionFactory` samples from a normal distribution centered on the reference value found in the Hamiltonian's molecule, clipped to an absolute range (here, at least 1.5 bohr) to avoid unphysical geometries.
Other distributions, such as :class:`~deepqmc.geom.zmatrix.CenteredUniformDistributionFactory`, sample uniformly within an offset around the reference value instead. A more elaborate example resampling both bond lengths and the bond angle of a water molecule is given on the :ref:`examples <examples>` page.

The dynamic sampler is then enabled with::

        $ deepqmc hydra.run.dir=workdir_transferable_continous task=train_transferable task.seed=42 ansatz=transpsiformer task.molecule_batch_size=1 hamil/mol=LiH task.mols.directory=null task/sampler_factory/nuc_sampler=LiH_zmat task/sampler_factory/elec_warp_fn=nn_elec_warp +task.sampler_factory.update_nuc_period=10

``update_nuc_period`` sets how often, in training steps, a new geometry is drawn.
Since it is not part of the default sampler configuration it needs to be added with the ``+`` prefix.
``elec_warp_fn=nn_elec_warp`` displaces each electron together with its nearest nucleus whenever the geometry changes, so that the electron walkers remain close to equilibrium and do not have to fully re-equilibrate from scratch after every geometry update.
If the geometry changes substantially between updates it can additionally help to set ``+task.sampler_factory.elec_equilibration_steps``, which runs a number of extra electron sampling steps immediately after each nuclear update, before that geometry's samples are used for training.

The trained wave function can now be evaluated on any geometry of interest similar to the previous example.
Typically interpolation within the training regime works very well, while extrapolation is much more challenging.
Note that, since the ansatz is not equivariant under rotations of the molecular geometry, the molecule should be aligned to a reference orientation before evaluation.

Ansatz
______

The hyperparameters of the training and the wave function ansatz are specified through hydra config files. Predefined ansatzes can be found in ``.../deepqmc/src/deepqmc/conf/ansatz`` and selected via::

    $ deepqmc ansatz=ferminet

The hyperparameters of such a predefined ansatz can also be overwritten at the command line::

    $ deepqmc ansatz=psiformer ansatz.omni_factory.gnn_factory.n_interactions=2

For convenience the configuration of the ``default`` ansatz is reproduced here:

.. literalinclude:: default.yaml
   :language: yaml
