.. _jax-api:

JAX API
=======

This is the documentation of the API of the JAX implementation of the :class:`deepqmc` package.

This implementation uses the `JAX library <https://github.com/google/jax>`_, the documentation for which can be found here:

- `JAX documentation <https://jax.readthedocs.io/en/latest>`_

.. automodule:: deepqmc.jax
   :exclude-members: train,evaluate

.. automodule:: deepqmc.jax.hamil

Haiku
-----

The JAX implementation uses the `haiku library <https://github.com/deepmind/dm-haiku>`_ to create, train and evaluate neural network models.
Some additional neural network functionality is implemented in the package and documented here.

.. automodule:: deepqmc.jax.hkext

Training
--------

.. automodule:: deepqmc.jax.train

.. automodule:: deepqmc.jax.fit

.. automodule:: deepqmc.jax.evaluate

.. automodule:: deepqmc.jax.equilibrate

Sampling
--------

.. automodule:: deepqmc.jax.sampling

Wave functions
--------------

.. autoclass:: deepqmc.jax.wf.WaveFunction

.. autoclass:: deepqmc.jax.wf.PauliNet

.. automodule:: deepqmc.jax.wf.paulinet.paulinet
   :exclude-members: PauliNet

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

