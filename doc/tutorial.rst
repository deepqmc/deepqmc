.. _tutorial:

Tutorial
========

This section describes the use of the high-level API of DeepQMC. For the lower-level API, consult directly the :ref:`api` documentation.

Create a molecule
-----------------

A molecule is represented by the :class:`~deepqmc.Molecule` class in DeepQMC. The easiest way to get started is to work with one of the predefined molecules::

   from deepqmc import Molecule

   mol = Molecule.from_name('LiH')

To get all available molecules use::

    >>> Molecule.all_names
    {'H2', 'Hn', 'CO2', 'H2+', 'bicyclobutane', 'B', 'Be', 'LiH', 'H'}

Molecule can be also crated from scratch, by specifying the nuclear coordinates and charges, and the total charge and spin multiplicity::

    mol = Molecule(  # LiH
        coords=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        charges=[3, 1],
        charge=0,
        spin=0,
    )

Create a wave function ansatz
-----------------------------

All wave function ansatzes are available in the :mod:`deepqmc.wf` subpackage. At the moment, the only available ansatz is :class:`~deepqmc.wf.PauliNet`. The PauliNet class has three different constructors, differing in how low-level they are. The high-level :meth:`PauliNet.from_hf` constructor has default parameters for everything except for the molecule, and it runs the underlying multireference Hartree--Fock calculation which provides the orbitals on which the ansatz is built::

    >>> from deepqmc.wf import PauliNet
    >>> net = PauliNet.from_hf(mol, cas=(4, 2)).cuda()
    converged SCF energy = -7.9846409186467
    CASSCF energy = -8.00207829274895
    CASCI E = -8.00207829274895  E(CI) = -1.09953139208267  S^2 = 0.0000000

All the parameters and their physical meaning are described in the :ref:`api` reference.

Optimize the ansatz
-------------------

The high-level :func:`~deepqmc.train` function is used to train the deep neural networks in the ansatz::

    >>> from deepqmc import train
    >>> train(net)
    training:   0%|▍       | 49/10000 [01:41<5:43:40,  2.07s/it, E=-8.0378(27)]

The terminal output shows only how far has the training progressed and the current estimate of the energy. More detailed monitoring of the training goes is available via `Tensorboard <https://www.tensorflow.org/tensorboard>`_. When :func:`~deepqmc.train` is called with an optional ``workdir`` argument, the training run creates a Tensorboard event file::

    >>> train(net, workdir='runs/01')

.. code:: none

    $ tensorboard --logdir runs/
    TensorFlow installation not found - running with reduced feature set.
    Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
    TensorBoard 2.1.0 at http://localhost:6007/ (Press CTRL+C to quit)

This launches a Tensorboard server which can be accessed via a web browser at the printed URL.

Get the energy
--------------

The rough estimate of the expectation value of the energy of a trained wave function can be obtained already from the training run. A rigorous estimation with a statistical sampling error can be obtained with the high-level :func:`~deepqmc.evaluate` function::

    >>> from deepqmc import evaluate
    >>> evaluate(net)
    evaluating: 100%|█████████| 500/500 [10:28<00:00,  1.26s/it, E=-8.07000(19)]
    {'energy': -8.07000108151436+/-0.000193955696684799}

As in the case of the training, the evaluation can be also monitored with Tensorboard.
