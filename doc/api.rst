.. _api:

API
===

The API is designed in such a way that entering it via the top-level :func:`deepqmc.train` and :func:`deepqmc.wf.PauliNet.from_hf` entry points enables complete specification of all parameters via nested dictionaries that can be encoded in a `TOML <https://github.com/toml-lang/toml>`_ file. The lower-level dictionaries (stored under ``*_kwargs`` keys) are gradually unwrapped and passed into lower-level API components.

All classes documented below with a Shape section are derived from :class:`torch.nn.Module` and are to be used as such.

Training
--------

.. automodule:: deepqmc

.. automodule:: deepqmc.fit

.. automodule:: deepqmc.sampling

Wave functions
--------------

.. module:: deepqmc.wf

.. autoclass:: deepqmc.wf.WaveFunction

PauliNet
~~~~~~~~

.. autoclass:: deepqmc.wf.PauliNet

.. automodule:: deepqmc.wf.paulinet
   :exclude-members: PauliNet
