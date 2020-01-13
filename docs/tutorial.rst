.. _tutorial:

Tutorial
========

This section describes the use of the high-level API of DeepQMC. For the lower-level API, go directly to the :ref:`api` documentation.

Create a molecule
-----------------

A molecule is represented by the :class:`~deepqmc.Molecule` class in DeepQMC. The easiest way to get started is to work with one of the predefined molecules::

   from deepqmc import Molecule

   mol = Molecule.from_name('LiH')

To get all available molecules use::

    >>> Molecule.all_names
    {'H2', 'Hn', 'CO2', 'H2+', 'bicyclobutane', 'B', 'Be', 'LiH', 'H'}

Molecule can be also crated from scratch, by specifying the nuclear coordinates and charges, and the total charge and spin multiplicity::

    mol = Molecule(  # H2+
        coords=[[0, 0, 0], [1.058, 0, 0]],
        charges=[1, 1],
        charge=1,
        spin=1,
    )

Create a wave function ansatz
-----------------------------

All wave function ansatzes are available in the :mod:`deepqmc.wf` package. At the moment, the only available ansatz is :class:`~deepqmc.wf.PauliNet`. The PauliNet class has three different constructors, differing in how low-level they are. The high-level :meth:`PauliNet.from_hf` constructor has default parameters for everything except for the molecule, and it runs the underlying multireference Hartree--Fock calculation which provides the orbitals on which the ansatz is built::

    >>> from deepqmc.wf import PauliNet
    >>> net = PauliNet.from_hf(mol)
    converged SCF energy = -7.9846409186467
    CASSCF energy = -8.00439006914284
    CASCI E = -8.00439006914284  E(CI) = -1.10200121173341  S^2 = 0.0000000

All the parameters and their physical meaning are described in the :ref:`api` reference.

Optimize the ansatz
-------------------

The high-level :func:`~deepqmc.train` function is used to train the deep neural networks in the ansatz::

    >>> from deepqmc import train
    >>> train(net)
    training:   0%|                                  | 0/10000 [00:00<?, ?it/s]
    sampling:  51%|█████████████             | 256/500 [00:26<00:25,  9.88it/s]

The terminal output shows only how far has the training progressed. All monitoring of how well the training goes is done via `Tensorboard <https://www.tensorflow.org/tensorboard>`_. The training run creates a Tensorboard event file in the `runs/` directory, which can be observed from Tensorboard::

    $ tensorboard --logdir runs/
    TensorFlow installation not found - running with reduced feature set.
    Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
    TensorBoard 2.1.0 at http://localhost:6007/ (Press CTRL+C to quit)

This launches a Tensorboard server which can be accessed in a web browser.
