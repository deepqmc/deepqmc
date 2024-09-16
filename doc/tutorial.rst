.. _tutorial:

Tutorial
========

This section exemplifies the use of the API of DeepQMC. For running calculations it is recommended to use the high-level command line API, which can be fully configured through hydra (see :ref:`cli <cli>`). For further information and a more detailed descriptions of the functions presented here consult the :ref:`api <api>` documentation and the accompanying software paper [Schaetzle23]_.

Create a molecule
-----------------

A molecule is represented by the :class:`~deepqmc.Molecule` class in DeepQMC. The easiest way to get started is to work with one of the predefined molecules::

   from deepqmc.molecule import Molecule

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

    from deepqmc.hamil import MolecularHamiltonian

    H = MolecularHamiltonian(mol=mol)

The Hamiltonian provides the local energy function for the evaluation of the energy expectation value, as well as an educated guess for initial electron configurations to start the sampling.

Create a wave function ansatz
-----------------------------

The neural network wave function ansatz is available in the :mod:`deepqmc.wf` submodule. A convenient way of initializing a wave function instance is to use a :mod:`hydra` config file. DeepQMC comes with config files for predefined wave functions (at ``../deepqmc/src/deepqmc/conf/ansatz``), however custom configurations may be used. Being a :mod:`haiku` module the ansatz has to be initialized inside a :func:`haiku.transform`::

    import os

    import haiku as hk
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    import deepqmc
    from deepqmc.app import instantiate_ansatz


    deepqmc_dir = os.path.dirname(deepqmc.__file__)
    config_dir = os.path.join(deepqmc_dir, 'conf/ansatz')

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='default')

    _ansatz = instantiate(cfg, _recursive_=True, _convert_='all')

    ansatz = instantiate_ansatz(H, _ansatz)

The hyperparameters and their physical meaning are described in the :ref:`api <api>` reference. The resulting ``ansatz`` object has two methods: ``ansatz.init`` can be used to initialize the ansatz parameters, while ``ansatz.apply`` evaluates the wave function.

Instantiate a sampler
---------------------

The variational Monte Carlo method requires sampling the probability density associated with the square of the wave function. A :class:`~deepqmc.sampling.Sampler` can be instantiated from a :class:`~deepqmc.hamil.MolecularHamiltonian` and a wave function. The instantiation is handled within the :func:`~deepqmc.train.train` wrapper of the training loop. :func:`~deepqmc.train.train` therefore accepts a sampler factory, that is a function that constructs a :class:`~deepqmc.sampling.Sampler`.::

    from deepqmc.sampling import initialize_sampling, MetropolisSampler, DecorrSampler, combine_samplers
    from functools import partial

    elec_sampler = partial(combine_samplers, samplers=[DecorrSampler(length=20), MetropolisSampler])
    sampler_factory = partial(initialize_sampling, elec_sampler=elec_sampler)

The above example combines the :class:`~deepqmc.sampling.DecorrSampler` and the :class:`~deepqmc.sampling.MetropolisSampler` to create a Metropolis-Hastings sampler that performs 20 decorrelating steps each time before returning the next set of samples.

Optimize the ansatz
-------------------

The high-level :func:`~deepqmc.train` function is used to train the deep neural networks in the ansatz. The train function takes a :class:`~deepqmc.hamil.MolecularHamiltonian`, a :class:`~deepqmc.wf.WaveFunction`, a sampler factory, and an :class:`~deepqmc.optimizer.Optimizer`. The recommended KFAC optimizer can be instantiated using :mod:`hydra`::

    config_dir = os.path.join(deepqmc_dir, 'conf/task/opt')

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='kfac')

    kfac = instantiate(cfg, _recursive_=True, _convert_='all')

The object ``kfac`` can now be passed as the ``opt`` argument to :func:`~deepqmc.train.train`, along with the requested number of training steps (``steps``), the number of electron position samples used in a training batch (``electron_batch_size``), and a seed (``seed``)::

    >>> from deepqmc.train import train
    >>> train(H, ansatz, kfac, sampler_factory, steps=10000, electron_batch_size=2000, seed=42)
    training:   0%|▋       | 102/10000 [01:00<23:01, 7.16it/s, E=-8.042(10)]

If the argument ``pretrain_steps`` is set, the ansatz is pretrained with respect to a Hartree-Fock or CASSCF baseline obtained with :mod:`pyscf`. For more details as well as further training hyperparameters consult the :ref:`api <api>` reference.

Optimizing electronic excited states
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DeepQMC can simultaneously optimize the lowest ``n`` electronic states of a molecule. By default, ``n`` is set to one, but it can be increased via the ``electronic_states`` argument to :func:`~deepqmc.train.train`. In order to efficiently pretrain for more than one electronic state, a CASSCF pretraining target should be used, with an active space that contains at least ``n`` states.

For a detailed description of the excited states methodology, see [Szabo24]_.

.. _logging:

Logging
-------

The terminal output shows only how far has the training progressed and the current estimate of the energy. More detailed monitoring of the training is available via `Tensorboard <https://www.tensorflow.org/tensorboard>`_. When :func:`~deepqmc.train.train` is called with an optional ``workdir`` argument, the training run creates a Tensorboard event file::

    train(H, ansatz, kfac, sampler_factory, steps=10000, electron_batch_size=2000, seed=42, workdir='runs/01')

.. code:: none

    $ tensorboard --logdir runs/
    TensorFlow installation not found - running with reduced feature set.
    Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
    TensorBoard 2.11.0 at http://localhost:6006/ (Press CTRL+C to quit)

This launches a Tensorboard server which can be accessed via a web browser at the printed URL.

Furthermore, several other quantities are dumped during the training to the ``workdir``. The ``training`` directory contains training checkpoints as well as an HDF5 file ``result.h5`` that holds the local energies throughout the training, an exponential moving average of the training energy and the values of the wave function at every iteration::

    >>> import h5py
    >>> with h5py.File('runs/01/training/result.h5') as f: print(f.keys(),f['local_energy'].keys())
    <KeysViewHDF5 ['local_energy']> <KeysViewHDF5 ['max', 'mean', 'min', 'samples', 'std']>

Additional observables can also be computed and logged during a run by specifying the ``observable_monitors`` argument to :func:`~deepqmc.train.train`. See also the :mod:`~deepqmc.observable` submodule for more details.

Evaluate the energy
-------------------

A rough estimate of the expectation value of the energy of a trained wave function can be obtained already from the local energies of the training run. A rigorous estimation of the energy expectation value up to the statistical sampling error can be obtained when evaluating the energy expectation value of the trained wavefunction without further optimization. This is achieved by passing a training checkpoint to the :func:`~deepqmc.train` function, and specifying the optimizer to be ``None``::

    >>> from deepqmc.log import CheckpointStore
    >>> step, train_state =CheckpointStore.load('runs/01/training/chkpt-10000.pt')
    >>> train(H, ansatz, None, sampler_factory, train_state=train_state, steps=500, electron_batch_size=2000, seed=42, workdir='runs/01')
    evaluating: 100%|█████████| 500/500 [01:20<00:00,  6.20it/s, E=-8.07000(19)]

The evaluation generates the same type of logs as the training, but writes to ``workdir/evaluation`` instead. The final energy can be read from the progress bar, the Tensorboard event file or computed from the local energies logged to the ``workdir/evaluation/result.h5`` file.

Pseudopotentials
-------------------

DeepQMC currently supports ``bfd`` [Burkatzki07]_ and ``ccECP`` [Bennett17]_ pseudopotentials, which can be enabled by passing the ``ecp_type`` argument to the Hamiltonian definition. This replaces a certain number of core electrons with a pseudopotential, reducing the total number of electrons explicitly treated and thus decreasing the computational cost. The pseudopotentials for all nuclei heavier than He in the molecule will be used if the argument ``ecp_type`` is passed. They can be turned off or on for individual nuclei by specifying ``pp_mask``, a boolean array with ``True`` (``False``) for each nucleus with pseudopotential turned on (off). The following example defines the Hamiltonian of a TiO molecule where the titanium core is replaced by a pseudopotential and the oxygen core is left unaffected::

    mol = Molecule(  # TiO
        coords=[[0.0, 0.0, 0.0], [1.668, 0.0, 0.0]],
        charges=[22, 8],
        charge=0,
        spin=2,
        unit='angstrom',
    )
    H = MolecularHamiltonian(mol=mol, ecp_type='ccECP', ecp_mask=[True,False], elec_std=0.1)

Systems containing heavier atoms sometimes tend to produce NaN errors. To avoid these issues, it was found useful to use a smaller variance for the initial distribution of electrons around the nuclei (via the ``elec_std`` argument) and a larger decorrelation length for sampling.
