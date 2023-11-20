import os
from typing import Sequence

import haiku as hk
import jax
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from jax import config

from deepqmc import MolecularHamiltonian, Molecule

config.update('jax_enable_x64', True)


@pytest.fixture(scope='session')
def helpers():
    return Helpers


class Helpers:
    @staticmethod
    def pytree_allclose(tree1, tree2):
        close = jax.tree_util.tree_map(jax.numpy.allclose, tree1, tree2)
        return jax.tree_util.tree_reduce(lambda x, y: x and y, close, True).item()

    @staticmethod
    def rng(seed=0):
        return jax.random.PRNGKey(seed)

    @staticmethod
    def flatten_pytree(d, parent_key='', sep=':'):
        if isinstance(d, Sequence):
            try:
                d = d._asdict()
            except AttributeError:
                d = {str(i): v for i, v in enumerate(d)}
        items = []
        for k, v in d.items():
            key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict) or isinstance(v, Sequence):
                items.extend(Helpers.flatten_pytree(v, parent_key=key, sep=sep).items())
            else:
                items.append((key, v))
        return dict(items)

    @staticmethod
    def mol(name='LiH'):
        return Molecule.from_name(name)

    @staticmethod
    def hamil(mol=None, pp_type=None, pp_mask=None):
        mol = mol or Helpers.mol()
        return MolecularHamiltonian(mol=mol, pp_type=pp_type, pp_mask=pp_mask)

    @staticmethod
    def R(hamil=None):
        hamil = hamil or Helpers.hamil()
        try:
            R = hamil.mol.coords
        except AttributeError:
            R = None
        return R

    @staticmethod
    def phys_conf(hamil=None, n=1, elec_std=1.0):
        hamil = hamil or Helpers.hamil()
        phys_conf = hamil.init_sample(Helpers.rng(), Helpers.R(hamil), n, elec_std)
        return phys_conf[0] if n == 1 else phys_conf

    @staticmethod
    def transform_model(model, *model_args, **model_kwargs):
        return hk.without_apply_rng(
            hk.transform(
                lambda *call_args: model(*model_args, **model_kwargs)(*call_args)
            )
        )

    @staticmethod
    def init_model(model, *args, seed=0):
        params = model.init(Helpers.rng(seed), *args)
        return params

    @staticmethod
    def create_ansatz(hamil=None):
        hamil = hamil or Helpers.hamil()
        _ansatz = Helpers.init_conf('ansatz')
        ansatz = Helpers.transform_model(_ansatz, hamil)
        params = Helpers.init_model(ansatz, Helpers.phys_conf(hamil))
        return ansatz, params

    @staticmethod
    def init_conf(config_name):
        with initialize_config_dir(
            version_base=None,
            config_dir=os.path.join(os.path.dirname(__file__), 'conf'),
        ):
            cfg = compose(config_name=config_name)
        return instantiate(cfg, _recursive_=True)
