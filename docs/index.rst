==================
Welcome to DeepQMC
==================

DeepQMC implements variational quantum Monte Carlo for electrons in molecules, using deep neural networks written in `PyTorch <https://pytorch.org>`_ as trial wave functions. Besides the core functionality, it contains implementations of the following ansatzes:

- PauliNet: https://arxiv.org/abs/1909.08423

Installing
==========

Install and update using `Pip <https://pip.pypa.io/en/stable/quickstart/>`_.

.. code-block:: bash

    pip install -U deepqmc[all]

A simple example
================

.. code-block::

    from deepqmc import Molecule, train
    from deepqmc.wf import PauliNet

    mol = Molecule.from_name('LiH')
    net = PauliNet.from_hf(mol)
    train(net)

API referencee
==============

.. toctree::

    api

.. toctree::
    :hidden:

    refs
