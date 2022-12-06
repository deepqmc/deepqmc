import haiku as hk
import jax
import pytest

from deepqmc import MolecularHamiltonian, Molecule
from deepqmc.wf import PauliNet, state_callback


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
        if isinstance(d, tuple):
            try:
                d = d._asdict()
            except AttributeError:
                d = {str(i): v for i, v in enumerate(d)}
        items = []
        for k, v in d.items():
            key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict) or isinstance(v, tuple):
                items.extend(Helpers.flatten_pytree(v, parent_key=key, sep=sep).items())
            else:
                items.append((key, v))
        return dict(items)

    @staticmethod
    def mol(name='LiH'):
        return Molecule.from_name(name)

    @staticmethod
    def hamil(mol=None):
        mol = mol or Helpers.mol()
        return MolecularHamiltonian(mol=mol)

    @staticmethod
    def rs(hamil=None, n=1):
        hamil = hamil or Helpers.hamil()
        rs = hamil.init_sample(Helpers.rng(), n)
        if n == 1:
            rs = rs[0]
        return rs

    @staticmethod
    def transform_model(model, *model_args, **model_kwargs):
        return hk.without_apply_rng(
            hk.transform_with_state(
                lambda *call_args: model(*model_args, **model_kwargs)(*call_args)
            )
        )

    @staticmethod
    def init_model(model, *args, seed=0, batch_dim=False):
        params, state = model.init(Helpers.rng(seed), *args)
        _, state = model.apply(params, state, *args)
        state, _ = state_callback(state, batch_dim=batch_dim)
        return params, state

    @staticmethod
    def create_paulinet(hamil=None, rs=None, init_model_kwargs=None, **kwargs):
        hamil = hamil or Helpers.hamil()
        return_rs = rs is None
        rs = rs or Helpers.rs()
        paulinet = Helpers.transform_model(PauliNet, hamil, **kwargs)
        params, state = Helpers.init_model(paulinet, rs, **(init_model_kwargs or {}))
        ret = (params, state, paulinet)
        if return_rs:
            ret += (rs,)
        return ret


def pytest_sessionstart(session):
    jax.config.update('jax_platform_name', 'cpu')
