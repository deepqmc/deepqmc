.. _api:

:tocdepth: 3

API
=======

This is the documentation of the API of the JAX implementation of the :class:`deepqmc` package.

This implementation uses the `JAX library <https://github.com/google/jax>`_, the documentation for which can be found here:

- `JAX documentation <https://jax.readthedocs.io/en/latest>`_

.. automodule:: deepqmc
   :exclude-members: train, MolecularHamiltonian, DecorrSampler, MetropolisSampler, ResampledSampler

.. automodule:: deepqmc.hamil

Haiku
-----

The JAX implementation uses the `haiku library <https://github.com/deepmind/dm-haiku>`_ to create, train and evaluate neural network models.
Some additional neural network functionality is implemented in the package and documented here.

.. automodule:: deepqmc.hkext

Training and evaluation
-----------------------

.. automodule:: deepqmc.train

.. automodule:: deepqmc.fit

Sampling
--------

.. autoclass:: deepqmc.sampling.Sampler

.. automodule:: deepqmc.sampling

Wave functions
--------------

.. autoclass:: deepqmc.wf.WaveFunction

.. autoclass:: deepqmc.wf.NeuralNetworkWaveFunction

.. automodule:: deepqmc.wf.nn_wave_function.nn_wave_function
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

Update Features
^^^^^^^^^^^^^^^

This submodule implements some common ways to compute update features for the node embeddings from the current node and edge embeddings.
Instances of the below classes are callable, they take as input the current node and edge representations, and output a list of update features
to be used for updating the node representations.

.. automodule:: deepqmc.gnn.update_features


.. toctree::
   :maxdepth: 2
