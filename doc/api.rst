.. _api:

:tocdepth: 3

API
=======

This is the documentation of the API of the :class:`deepqmc` package.

This implementation uses the `JAX library <https://github.com/google/jax>`_, the documentation for which can be found here:

- `JAX documentation <https://jax.readthedocs.io/en/latest>`_

.. automodule:: deepqmc.molecule

.. automodule:: deepqmc.hamil


Haiku
-----

The JAX implementation uses the `haiku library <https://github.com/deepmind/dm-haiku>`_ to create, train and evaluate neural network models.
Some additional neural network functionality is implemented in the package and documented here.

.. automodule:: deepqmc.hkext

Training and evaluation
-----------------------

.. autoclass:: deepqmc.types.TrainState

.. automodule:: deepqmc.train

.. automodule:: deepqmc.fit

Sampling
--------

.. autoclass:: deepqmc.types.SamplerState

.. autoclass:: deepqmc.sampling.base.ElectronSampler

.. automodule:: deepqmc.sampling
   :exclude-members: nuclei_sampling.ZMatrixSampler

Optimizers
----------

.. automodule:: deepqmc.optimizer

Wave functions
--------------

.. autoclass:: deepqmc.types.Psi

.. autoclass:: deepqmc.types.WaveFunction

.. autoclass:: deepqmc.types.ParametrizedWaveFunction

.. autoclass:: deepqmc.types.Ansatz

.. autoclass:: deepqmc.wf.NeuralNetworkWaveFunction

.. automodule:: deepqmc.wf.nn_wave_function
   :exclude-members: NeuralNetworkWaveFunction

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
Instances of the below classes are callable, they take as input the current node and edge representations, and output a list of update features
to be used for updating the node representations.

.. automodule:: deepqmc.gnn.update_features
   :exclude-members: CombinedNodeAttentionUpdateFeature

.. toctree::
   :maxdepth: 2

.. autoclass:: deepqmc.gnn.utils.NodeEdgeMapping

Excited electronic states
-------------------------

This section documents the API used to treat electronic excited states.

.. autoclass:: deepqmc.sampling.combined_samplers.MultiElectronicStateSampler

.. automodule:: deepqmc.loss.overlap

Custom data types and type aliases
----------------------------------

In order to facilitate the use of the API and enable type checking DeepQMC implements a range of custom types and type aliases.

A combination of electron and nuclei positions is assigned the PhysicalConfiguration type.

.. autoclass:: deepqmc.types.PhysicalConfiguration

Wave function parameters are denoted with the Params type.

.. autoclass:: deepqmc.types.Params

Auxiliary data need for the evaluation of the loss comes as a DataDict type.

.. autoclass:: deepqmc.types.DataDict

Statistics obtained during training or evaluation use the Stats type.

.. autoclass:: deepqmc.types.Stats

Data for the evaluation of the training loss is bundeld as a Batch type.

.. autoclass:: deepqmc.types.Batch
