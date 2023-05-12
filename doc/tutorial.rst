.. _tutorial:

Tutorial
========

This section exemplifies the use of the API of DeepQMC. For the high-level command line API see :ref:`cli <cli>`. For further information and a more detailed descriptions of the functions presented here consult the :ref:`api <api>` documentation.

Create a molecule
-----------------

A molecule is represented by the :class:`~deepqmc.Molecule` class in DeepQMC. The easiest way to get started is to work with one of the predefined molecules::

   from deepqmc import Molecule

   mol = Molecule.from_name('LiH')

To get all available predefined molecules use::

    >>> Molecule.all_names
    {'B', 'B2', 'Be', ..., bicyclobutane'}

A :class:`~deepqmc.Molecule` can be also created from scratch by specifying the nuclear coordinates and charges, as well as the total charge and spin multiplicity::

    mol = Molecule(  # LiH
        coords=[[0.0, 0.0, 0.0], [3.015, 0.0, 0.0]],
        charges=[3, 1],
        charge=0,
        spin=0,
        unit='bohr',
    )

Create the molecular Hamiltonian
--------------------------------

From the molecule the :class:`~deepqmc.hamil.MolecularHamiltonian` is constructed::
        
    from deepqmc import MolecularHamiltonian

    H = MolecularHamiltonian(mol=mol)

The Hamiltonian provides the local energy function for the evaluation of the energy expectation value, as well as an educated guess for initial electron configurations to start the sampling.

Create a wave function ansatz
-----------------------------

The neural network wave function ansatz is available in the :mod:`deepqmc.wf` subpackage. A convinient way of initializing a wave function instance is to use a :mod:`hydra` config file. DeepQMC comes with config files for predefined wave functions (at ``../deepqmc/src/deepqmc/conf/ansatz``), however custom configurations may be used. Being a :mod:`haiku` module the ansatz has to be initialized inside a :func:`haiku.transform`::

    import os

    import deepqmc
    import haiku as hk
    from deepqmc.wf import NeuralNetworkWaveFunction
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    deepqmc_dir = os.path.dirname(deepqmc.__file__)
    config_dir = os.path.join(deepqmc_dir, 'conf/ansatz')

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='paulinet')

    _ansatz = instantiate(cfg, _recursive_=True)


    @hk.without_apply_rng
    @hk.transform
    def ansatz(phys_conf, return_mos=False):
        return _ansatz(H)(phys_conf, return_mos=return_mos)

The hyperparameters and their physical meaning are described in the :ref:`api <api>` reference.

Instantiate a sampler
---------------------

The variational Monte Carlo method requires sampling the propability density associated with the square of the wave function. A :class:`~deepqmc.sampling.Sampler` can be instantiated from a :class:`~deepqmc.wf.WaveFunction`::

    from deepqmc.sampling import chain, MetropolisSampler, DecorrSampler

    sampler = chain(DecorrSampler(length=20),MetropolisSampler(H))

Different samplers can be chained together via the :func:`~deepqmc.sampling.chain` command.

Optimize the ansatz
-------------------

The high-level :func:`~deepqmc.train` function is used to train the deep neural networks in the ansatz. The train function takes a :class:`~deepqmc.hamil.MolecularHamiltonian`, a :class:`~deepqmc.wf.WaveFunction` and a :class:`~deepqmc.sampling.Sampler`. Further necessary arguments are an optimizer (``opt``), the number of training steps (``steps``), the number of samples used in a training batch (``sample_size``), and a seed (``seed``)::

    >>> from deepqmc import train
    >>> train(H, net, 'kfac', sampler, steps=10000, sample_size=2000, seed=42)
    training:   0%|▋       | 102/10000 [01:00<23:01, 7.16it/s, E=-8.042(10)]

If the argument ``pretrain_steps`` is set, the ansatz is pretrained with respect to a Hartree-Fock or CASSCF baseline obtained with :mod:`pyscf`. For more details as well as further training hyperparameters consult the :ref:`api <api>` reference.

Logging
-------

The terminal output shows only how far has the training progressed and the current estimate of the energy. More detailed monitoring of the training is available via `Tensorboard <https://www.tensorflow.org/tensorboard>`_. When :func:`~deepqmc.train` is called with an optional ``workdir`` argument, the training run creates a Tensorboard event file::

    >>> train(net, workdir='runs/01')

.. code:: none

    $ tensorboard --logdir runs/
    TensorFlow installation not found - running with reduced feature set.
    Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
    TensorBoard 2.11.0 at http://localhost:6006/ (Press CTRL+C to quit)

This launches a Tensorboard server which can be accessed via a web browser at the printed URL.

Furthermore the training run is logged to the ``workdir``. The ``training`` directory contains training checkpoints as well as an hdf5 file ``result.h5`` that holds the local energies throughout the training, an exponential moving average of the training energy and the values of the wave function at every iteration::

    >>> import h5py
    >>> with h5py.File('workdir/training/result.h5') as f: print(f.keys())
    <KeysViewHDF5 ['E_ewm', 'E_loc', 'log_psi', 'sign_psi']>

Evaluate the energy
-------------------

A rough estimate of the expectation value of the energy of a trained wave function can be obtained already from the local energies of the training run. A rigorous estimation of the energy expectation value up to the statistical sampling error can be obtained when evaluating the energy expectation value of the trained wavefunction without further optimization. This is achieved by passing a training checkpoint is passed to the :func:`~deepqmc.train` function, but the optimizer is specified to be ``None``:

    >>> import jax.numpy as jnp
    >>> train_state = jnp.load('workdir/training/chkpt-10000.pt',allow_pickle=True)
    >>> train(H, net, None, sampler, train_state=train_state, steps=500, sample_size=2000, seed=42)
    evaluating: 100%|█████████| 500/500 [01:20<00:00,  6.20it/s, E=-8.07000(19)]

The evaluation generates the same type of logs as the training, but writes to ``workdir/evaluation`` instead. The final energy can be read from the progress bar, the Tensorboard event file or computed from the local enregies in the hdf5 file respectively.
