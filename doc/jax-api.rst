.. _jax-api:

JAX API
=======

This is the documentation of the API of the JAX implementation of the :class:`deepqmc` package.

This implementation uses the `JAX library <https://github.com/google/jax>`_, the documentation for which can be found here:

- `JAX documentation <https://jax.readthedocs.io/en/latest>`_

.. automodule:: deepqmc.jax
   :exclude-members: train,evaluate

.. automodule:: deepqmc.jax.hamil

Training
--------

.. automodule:: deepqmc.jax.train

.. automodule:: deepqmc.jax.fit

Evaluation
----------

.. automodule:: deepqmc.jax.evaluate

Sampling
--------

.. automodule:: deepqmc.jax.sampling

Wave functions
--------------

.. autoclass:: deepqmc.jax.wf.WaveFunction

Graph neural networks
---------------------

A graph neural network is the most important component of the PauliNet Ansatz.
This module implements the most useful GNNs used in deep QMC.

Graphs
~~~~~~

This submodule implements the basic functionality for working with (truncated) graphs.

.. automodule:: deepqmc.jax.gnn.graph

.. automodule:: deepqmc.jax.gnn.edge_features

GNNs
~~~~

This submodule provides the base classes that all GNN instances should inherit from.

.. automodule:: deepqmc.jax.gnn.gnn

SchNet
~~~~~~

This submodule defines the SchNet architecutre adapted for graphs of nuclei and electrons.

.. automodule:: deepqmc.jax.gnn.schnet

