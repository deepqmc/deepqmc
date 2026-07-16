.. _examples:

Examples
========

This page contains a few fully worked examples that run an experiment via the CLI and then analyze the results with :mod:`deepqmc.postprocess`.
They are executed by the authors and checked in with their outputs already saved, since actually running a DeepQMC training during the documentation build is not feasible.
To run one yourself, install the ``notebooks`` extra and register a kernel for it::

    $ pip install -e .[notebooks]
    $ python -m ipykernel install --user --name deepqmc --display-name "DeepQMC"

.. toctree::
   :maxdepth: 1

   examples/ground_state_lih
