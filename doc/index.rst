==================
Welcome to DeepQMC
==================

Welcome to the DeepQMC documentation. Start with :ref:`installation <installation>` and run your first experiments via the :ref:`cli <cli>`. Learn more about the low level structure of the package by continuing with the :ref:`tutorial <tutorial>`, which walks you through the basic usage of the package and exposes the key objects, such as the wave function ansatz, the sampler and the optimizer. Finally, detailed technical reference can be found in the :ref:`api <api>` section.

DeepQMC is based on `JAX <https://github.com/google/jax>`_ and `Haiku <https://github.com/deepmind/dm-haiku>`_, the documentation for which can be found here:

- `JAX documentation <https://jax.readthedocs.io/en/latest>`_
- `Haiku documentation <https://dm-haiku.readthedocs.io/en/latest>`_

For an introduction to the methodology of variational Monte Carlo with neural network trial wave functions, details on the deep-learning Ansätze and applications of the DeepQMC program package we refer to the accompanying software paper [Schaetzle23]_.

DeepQMC implements the optimization of electronic excited states with a penalty method as described in [Szabo24]_ and introduced in [Entwistle23]_.


User guide
==========

.. toctree::
    :maxdepth: 1

    installation
    cli
    tutorial

API reference
=============

.. toctree::
    :maxdepth: 2

    api

.. toctree::
    :hidden:

    refs
