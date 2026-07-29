.. _api:

:tocdepth: 3

API
=======

This is the documentation of the API of the ``deepqmc`` package.

This implementation of ``deepqmc`` uses the `JAX library <https://github.com/google/jax>`_.
Neural network wave function in ``deepqmc`` are build using `haiku <https://github.com/deepmind/dm-haiku>`_.
The documentation for which can be found here:

- `JAX documentation <https://jax.readthedocs.io/en/latest>`_
- `Haiku documentation <https://dm-haiku.readthedocs.io/en/latest/>`_


Molecules and Hamiltonians
--------------------------

.. automodule:: deepqmc.molecule

Unit conversions
~~~~~~~~~~~~~~~~

Physical quantities such as coordinates or energies are represented internally in
atomic units (bohr, hartree). The functions below convert to and from other common
units, e.g. to specify a :class:`~deepqmc.molecule.Molecule`'s coordinates in
angstrom, or to report a computed energy in electronvolts or kcal/mol.

.. automodule:: deepqmc.units

.. autoclass:: deepqmc.hamil.Hamiltonian
   :members:

.. autoclass:: deepqmc.hamil.MolecularHamiltonian

Laplacian evaluation
~~~~~~~~~~~~~~~~~~~~

:class:`~deepqmc.hamil.MolecularHamiltonian` computes the kinetic-energy term of the
Hamiltonian via its ``laplacian_factory`` argument.

.. autoclass:: deepqmc.physics.LaplacianFactory
   :members:

.. autofunction:: deepqmc.physics.reverse_forward_laplacian

Potentials
----------

:class:`~deepqmc.hamil.MolecularHamiltonian` represents the electron-nucleus
interaction as a :class:`~deepqmc.physics.Potential`, selected via its ``ecp_type``
argument: plain Coulomb attraction by default, or, if ``ecp_type`` is set, a
Gaussian-type effective core potential (as implemented in :mod:`pyscf`) or a local
pseudo-Hamiltonian.

.. autoclass:: deepqmc.physics.Potential
   :members:

.. autoclass:: deepqmc.physics.NuclearCoulombPotential

Effective core potentials
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: deepqmc.ecp.gaussian_type_ecp.GaussianTypeECP

Pseudo-Hamiltonians
~~~~~~~~~~~~~~~~~~~~

A local alternative to effective core potentials, avoiding the nonlocal potential's
stochastic quadrature evaluation, currently available for P, S, Cl, Cr, Mn, Fe, Co, Ni, Cu
and Zn.

.. autoclass:: deepqmc.ecp.pseudo_hamiltonian.PseudoHamiltonian

Training and evaluation
-----------------------

.. autoclass:: deepqmc.types.TrainState

.. automodule:: deepqmc.train

.. automodule:: deepqmc.fit

Exceptions
~~~~~~~~~~

Numerical instabilities encountered during training (NaNs, sudden energy blowups) are
signalled with custom exceptions, which :func:`~deepqmc.train.train` catches internally
to restart from the last checkpoint.

.. automodule:: deepqmc.exceptions

Exponential moving averages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Running means and variances of training statistics (e.g. the energy) are tracked with
an exponentially weighted moving average, whose per-step weight adaptively decays as
more observations are folded in.

.. autoclass:: deepqmc.ewm.EWMState

.. autofunction:: deepqmc.ewm.init_ewm

.. autofunction:: deepqmc.ewm.init_multi_mol_multi_state_ewm

Application entry points
~~~~~~~~~~~~~~~~~~~~~~~~~

The :mod:`deepqmc.app` module wires :func:`~deepqmc.train.train` up to the
:mod:`hydra`-configured command line application (see :ref:`cli <cli>`), and provides
the function used to instantiate an ansatz outside of a full training run (see the
:ref:`tutorial <tutorial>`).

.. autodata:: deepqmc.types.AnsatzFactory

.. autofunction:: deepqmc.app.instantiate_ansatz

.. autofunction:: deepqmc.app.train_from_factories

.. autofunction:: deepqmc.app.train_from_checkpoint

.. autofunction:: deepqmc.app.read_molecules

Pretraining
-----------

If the argument ``pretrain_steps`` of :func:`~deepqmc.train.train` is set, the wave
function ansatz is pretrained to match a Hartree-Fock or CASSCF baseline obtained with
:mod:`pyscf`, before the variational optimization starts.

.. autofunction:: deepqmc.pretrain.pretrain

PySCF baseline
~~~~~~~~~~~~~~

The baseline (MC-)SCF solution and the Gaussian basis it is expressed in are computed
with the following helpers, and bundled into the ``dataset`` consumed by
:func:`~deepqmc.pretrain.pretrain`.

.. autofunction:: deepqmc.pretrain.pyscfext.compute_scf_solution

.. autofunction:: deepqmc.pretrain.pyscfext.pyscf_from_hamil

.. autofunction:: deepqmc.pretrain.pyscfext.pyscf_from_chkfile

.. autofunction:: deepqmc.pretrain.pyscfext.confs_from_mc

.. autoclass:: deepqmc.pretrain.gto.GTOBasis

Loss functions
--------------

.. autoclass:: deepqmc.loss.LossFunction
   :members:

.. autoclass:: deepqmc.loss.LossFunctionFactory
   :members:

.. autoclass:: deepqmc.loss.LossAndGradFunction
   :members:

.. autofunction:: deepqmc.loss.create_loss_fn

Energy loss
~~~~~~~~~~~

.. automodule:: deepqmc.loss.energy

Overlap loss
~~~~~~~~~~~~

.. automodule:: deepqmc.loss.overlap

Spin loss
~~~~~~~~~

.. automodule:: deepqmc.loss.spin

Clipping
~~~~~~~~

.. automodule:: deepqmc.loss.clip

Sampling
--------

.. autodata:: deepqmc.types.SamplerState

.. autodata:: deepqmc.types.SamplerFactory

Electron samplers
~~~~~~~~~~~~~~~~~

.. autoclass:: deepqmc.sampling.base.ElectronSampler

.. autoclass:: deepqmc.sampling.MetropolisSampler

.. autoclass:: deepqmc.sampling.LangevinSampler

.. autoclass:: deepqmc.sampling.electron_samplers.OppositeSpinExchangeSampler

.. autoclass:: deepqmc.sampling.DecorrSampler

.. autofunction:: deepqmc.sampling.combine_samplers


Electron sample initializers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: deepqmc.sampling.electron_sample_initializers.ElectronSampleInitializer

.. autoclass:: deepqmc.sampling.electron_sample_initializers.AtomCenteredDistribution

.. autoclass:: deepqmc.sampling.electron_sample_initializers.SingleGaussianDistribution

.. autoclass:: deepqmc.sampling.electron_sample_initializers.ShellBasedDistribution

.. autoclass:: deepqmc.sampling.electron_sample_initializers.AtomCenteredElectronInitializer

Nuclei samplers
~~~~~~~~~~~~~~~

.. autoclass:: deepqmc.sampling.base.NucleiSampler
   :members:

.. autoclass:: deepqmc.sampling.base.ElectronWarp
   :members:

.. autoclass:: deepqmc.sampling.nuclei_samplers.IdleNucleiSampler

.. autoclass:: deepqmc.sampling.nuclei_samplers.ConstraintNucleiSampler

.. autoclass:: deepqmc.sampling.nuclei_samplers.PermutationNucleiSampler

.. autoclass:: deepqmc.sampling.nuclei_samplers.ZMatrixSampler

.. autofunction:: deepqmc.sampling.nuclei_samplers.no_elec_warp

.. autofunction:: deepqmc.sampling.nuclei_samplers.nn_elec_warp

.. autofunction:: deepqmc.sampling.nuclei_samplers.fn_elec_warp


Multi state and multi geometry samplers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: deepqmc.sampling.MoleculeIdxSampler

.. autoclass:: deepqmc.sampling.combined_samplers.MultiElectronicStateSampler

.. autoclass:: deepqmc.sampling.MultiNuclearGeometrySampler

Setting up and equilibrating sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: deepqmc.sampling.initialize_sampling

.. autofunction:: deepqmc.sampling.initialize_sampler_state

.. autofunction:: deepqmc.sampling.equilibrate

Nuclear geometry
-----------------

This module implements the internal-coordinate machinery underlying nuclear geometry
sampling: functions to compute bond lengths, angles and dihedral angles, coordinate
transforms between Cartesian and internal coordinates, and Z-matrices. It is used e.g.
by :class:`~deepqmc.sampling.nuclei_samplers.ConstraintNucleiSampler` and
:class:`~deepqmc.sampling.nuclei_samplers.ZMatrixSampler` to sample or constrain
nuclear positions in a coordinate system other than Cartesian.

.. autofunction:: deepqmc.geom.distance

.. autofunction:: deepqmc.geom.angle

.. autofunction:: deepqmc.geom.dihedral

Coordinate transforms
~~~~~~~~~~~~~~~~~~~~~~

A coordinate transform maps Cartesian nuclear coordinates to another coordinate
representation. Invertible coordinate transforms can additionally map coordinates back
to Cartesian space, which is required to use them as the ``coordinate_transform``
argument of :class:`~deepqmc.sampling.nuclei_samplers.ConstraintNucleiSampler`.

.. autoclass:: deepqmc.geom.coordinate_transform.CoordinateTransform
   :members:

.. autoclass:: deepqmc.geom.coordinate_transform.InvertibleCoordinateTransform
   :members:

.. autoclass:: deepqmc.geom.coordinate_transform.CartesianCoordinateTransform

.. autoclass:: deepqmc.geom.coordinate_transform.ZMatrixCoordinateTransform

.. autoclass:: deepqmc.geom.coordinate_transform.RedundantInternalCoordinateTransform

.. autoclass:: deepqmc.geom.coordinate_transform.GeneralCoordinateTransform

.. autoclass:: deepqmc.geom.coordinate_transform.SubsetCoordinateTransform

Z-matrices
~~~~~~~~~~

A Z-matrix specifies a molecular geometry in terms of bond lengths, bond angles and
dihedral angles rather than Cartesian coordinates, each defined relative to previously
placed atoms. :class:`~deepqmc.geom.zmatrix.ConcreteZMatrixTemplate` defines the
connectivity of such a Z-matrix and can be turned into a
:class:`~deepqmc.geom.coordinate_transform.ZMatrixCoordinateTransform`, while
:class:`~deepqmc.geom.zmatrix.StochasticZMatrixTemplate` additionally attaches a noise
distribution to each entry, allowing new geometries to be sampled directly in internal
coordinates (as used by
:class:`~deepqmc.sampling.nuclei_samplers.ZMatrixSampler`).

.. autoclass:: deepqmc.geom.zmatrix.ConcreteZMatrixTemplate
   :members:
   :inherited-members:

.. autoclass:: deepqmc.geom.zmatrix.ConcreteZMatrix
   :members:

.. autoclass:: deepqmc.geom.zmatrix.StochasticZMatrixTemplate
   :members:
   :inherited-members:

.. autoclass:: deepqmc.geom.zmatrix.StochasticZMatrix
   :members:

Distributions for stochastic Z-matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These classes implement the ``DistributionFactory`` protocol, and are used to specify
the noise distribution of the individual bond lengths, angles and dihedrals of a
:class:`~deepqmc.geom.zmatrix.StochasticZMatrixTemplate`.

.. autoclass:: deepqmc.geom.zmatrix.stochastic.DistributionFactory
   :members:

.. autoclass:: deepqmc.geom.zmatrix.UniformDistributionFactory

.. autoclass:: deepqmc.geom.zmatrix.CenteredUniformDistributionFactory

.. autoclass:: deepqmc.geom.zmatrix.RadiallyUniformDistributionFactory

.. autoclass:: deepqmc.geom.zmatrix.CenteredRadiallyUniformDistributionFactory

.. autoclass:: deepqmc.geom.zmatrix.ClippedNormalDistributionFactory

.. autoclass:: deepqmc.geom.zmatrix.ClippedAsymmetricNormalDistributionFactory

.. autoclass:: deepqmc.geom.zmatrix.DeltaDistributionFactory


Optimizers
----------

.. autodata:: deepqmc.types.OptState

.. autodata:: deepqmc.types.OptimizerFactory

.. autoclass:: deepqmc.optimizer.Optimizer
   :members:

.. autoclass:: deepqmc.optimizer.NoOptimizer

.. autoclass:: deepqmc.optimizer.OptaxOptimizer

.. autoclass:: deepqmc.optimizer.KFACOptimizer

.. autofunction:: deepqmc.optimizer.merge_states

.. autofunction:: deepqmc.kfacext.batch_size_extractor

Schedules
~~~~~~~~~

Learning rate and damping schedules for use with
:class:`~deepqmc.optimizer.KFACOptimizer` or any :mod:`optax` optimizer.

.. autofunction:: deepqmc.utils.InverseSchedule

.. autofunction:: deepqmc.utils.ConstantSchedule

Wave functions
--------------

.. autoclass:: deepqmc.types.Psi

.. autodata:: deepqmc.types.WaveFunction

.. autodata:: deepqmc.types.ParametrizedWaveFunction

.. autoclass:: deepqmc.types.Ansatz

.. autoclass:: deepqmc.wf.NeuralNetworkWaveFunction

.. automodule:: deepqmc.wf.nn_wave_function
   :exclude-members: NeuralNetworkWaveFunction

Omni-net, envelopes and cusp corrections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~deepqmc.wf.nn_wave_function.NeuralNetworkWaveFunction` combines a GNN
embedding with Jastrow, backflow, envelope and electronic/nuclear cusp factors, each
configurable via a corresponding factory argument (``omni_factory``, ``envelope``,
``cusp_electrons`` and ``cusp_nuclei``).

.. automodule:: deepqmc.wf.omni

Envelopes
^^^^^^^^^

.. automodule:: deepqmc.wf.env

Cusp corrections
^^^^^^^^^^^^^^^^^

.. automodule:: deepqmc.wf.cusp

Graph neural networks
---------------------

A graph neural network is the most important component of the neural network wave function ansatz.
This module implements a general gnn framework, that can be configured to obtain a variety of different ansatzes.

Graphs
~~~~~~

This submodule implements the basic functionality for working with graphs.

.. automodule:: deepqmc.gnn.graph

.. automodule:: deepqmc.gnn.edge_features

Electron GNN
~~~~~~~~~~~~

This submodule provides the ElectronGNN architecture for defining neural network
parametrized functions acting on graphs of electrons and nuclei.

.. automodule:: deepqmc.gnn.electron_gnn
   :exclude-members: PermutationInvariantEmbedding

Update Features
^^^^^^^^^^^^^^^

This submodule implements some common ways to compute update features for the node embeddings from the current node and edge embeddings.
Instances of the below classes are callable, they take as input the current node and edge representations, and output a list of update features to be used for updating the node representations.

.. automodule:: deepqmc.gnn.update_features
   :exclude-members: CombinedNodeAttentionUpdateFeature

.. autoclass:: deepqmc.gnn.utils.NodeEdgeMapping

Haiku
~~~~~

Some additional neural network functionality is implemented in the package and documented here.

.. automodule:: deepqmc.hkext

Observables
-----------

.. autoclass:: deepqmc.observable.ObservableMonitor

Property monitors
~~~~~~~~~~~~~~~~~

.. autoclass:: deepqmc.observable.EnergyMonitor

.. autoclass:: deepqmc.observable.SpinMonitor

.. autoclass:: deepqmc.observable.WaveFunctionMonitor

.. autoclass:: deepqmc.observable.PsiRatioMonitor

.. autoclass:: deepqmc.observable.ElectronPositionMonitor

.. autoclass:: deepqmc.observable.NuclearPositionMonitor

.. autoclass:: deepqmc.observable.OscillatorStrengthMonitor

Oscillator strength
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: deepqmc.oscillator_strength.compute_oscillator_strength

Force monitors
~~~~~~~~~~~~~~

.. autoclass:: deepqmc.observable.BareForceMonitor

.. autoclass:: deepqmc.observable.BareForceAntiMonitor

.. autoclass:: deepqmc.observable.ACZVForceMonitor

.. autoclass:: deepqmc.observable.ACZVZBForceMonitor

.. autoclass:: deepqmc.observable.ACZBForceMonitor

.. autoclass:: deepqmc.observable.ACZVQForceMonitor

.. autoclass:: deepqmc.observable.ACZVQForceAntiMonitor

.. autoclass:: deepqmc.observable.ACZVZBQForceMonitor

.. autoclass:: deepqmc.observable.ACZVQZBForceMonitor

.. autoclass:: deepqmc.observable.FiniteDifferenceForceMonitor

Force estimators
~~~~~~~~~~~~~~~~

The force monitors above wrap the following lower-level estimator functions, which
construct the Hellmann-Feynman and finite-difference force evaluators used during
training and evaluation. They can also be called directly for postprocessing analysis.

.. autofunction:: deepqmc.force.evaluate_hf_force_bare

.. autofunction:: deepqmc.force.evaluate_hf_force_ac_zv

.. autofunction:: deepqmc.force.evaluate_hf_force_ac_zvzb

.. autofunction:: deepqmc.force.evaluate_hf_force_ac_zb

.. autofunction:: deepqmc.force.evaluate_hf_force_ac_zvq

.. autofunction:: deepqmc.force.evaluate_hf_force_ac_zvzbq

.. autofunction:: deepqmc.force.evaluate_hf_force_ac_zvqzb

.. autofunction:: deepqmc.force.evaluate_finite_difference_force

.. autofunction:: deepqmc.force.antithetic_wrapper

Multi-device execution
-----------------------

DeepQMC parallelizes training and evaluation across the available GPUs, see
:ref:`cli:Execution on multiple GPUs`. The module below implements the low-level
multi-device and multi-host primitives used throughout the package, and is useful when
implementing custom :class:`~deepqmc.observable.ObservableMonitor`,
:class:`~deepqmc.loss.LossFunction`, or other extensions that need to be aware of the
underlying device parallelism.

.. automodule:: deepqmc.parallel
   :exclude-members: get_process_count, get_process_index, maybe_init_multi_host

Logging
-------

.. automodule:: deepqmc.log

Postprocessing
--------------

The :mod:`deepqmc.postprocess` package collects utilities for analyzing a finished
training or evaluation run: reinstantiating a trained ansatz from a checkpoint,
reading logged observables from a workdir, and estimating their Monte Carlo sampling
error.

Checkpoints and ansatzes
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: deepqmc.postprocess.checkpoint_utils.load_parameters

.. autofunction:: deepqmc.postprocess.checkpoint_utils.phys_conf_from_checkpoint

.. autofunction:: deepqmc.postprocess.ansatz_utils.instantiate_predefined_ansatz

.. autofunction:: deepqmc.postprocess.ansatz_utils.instantiate_wf_from_checkpoint

Reading workdirs
~~~~~~~~~~~~~~~~

.. autofunction:: deepqmc.postprocess.workdir.read_workdir

.. autofunction:: deepqmc.postprocess.workdir.read_and_reshape_result

.. autofunction:: deepqmc.postprocess.workdir.read_and_convert_result

.. autofunction:: deepqmc.postprocess.workdir.convert_to_per_molecule_format

.. autofunction:: deepqmc.postprocess.workdir.gather_electron_axis

.. autofunction:: deepqmc.postprocess.workdir.last_checkpoint_iteration

.. autofunction:: deepqmc.postprocess.workdir.read_average_iteration_time

Monte Carlo statistics
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: deepqmc.postprocess.mc_utils.sampling_error

.. autofunction:: deepqmc.postprocess.mc_utils.clipped_mean_and_sampling_error

.. autofunction:: deepqmc.postprocess.mc_utils.clipped_batch_mean_and_std

Custom data types and type aliases
----------------------------------

In order to facilitate the use of the API and enable type checking DeepQMC implements a range of custom types and type aliases.

A combination of electron and nuclei positions, with an molecular geometry index is assigned the PhysicalConfiguration type.

.. autoclass:: deepqmc.types.PhysicalConfiguration

Wave function parameters are denoted with the Params type.

.. autodata:: deepqmc.types.Params

Auxiliary data need for the evaluation of the loss comes as a DataDict type.

.. autodata:: deepqmc.types.DataDict

Statistics obtained during training or evaluation use the Stats type.

.. autodata:: deepqmc.types.Stats

Data for the evaluation of the training loss is bundeld as a Batch type.

.. autodata:: deepqmc.types.Batch

Generating random numbers in jax requires an rng key, which is declared a KeyArray,

.. autodata:: deepqmc.types.KeyArray

Evaluated local energies are stored as an array of the Energy type.

.. autodata:: deepqmc.types.Energy

Sample importance weights are stored as an array of the Weight type.

.. autodata:: deepqmc.types.Weight
