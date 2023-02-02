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
    def mol(name='LiH', pp_type=None):
        return Molecule.from_name(name, pp_type=pp_type)

    @staticmethod
    def hamil(mol=None):
        mol = mol or Helpers.mol()
        return MolecularHamiltonian(mol=mol)

    @staticmethod
    def R(mol_name='LiH'):
        return Molecule.default_coords_from_name(mol_name)

    @staticmethod
    def phys_conf(hamil=None, R=None, n=1, elec_std=1.0):
        hamil = hamil or Helpers.hamil()
        if R is None:
            R = Helpers.R()
        phys_conf = hamil.init_sample(Helpers.rng(), R, n, elec_std)
        return phys_conf[0] if n == 1 else phys_conf

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
    def create_paulinet(
        hamil=None,
        phys_conf=None,
        R=None,
        init_model_kwargs=None,
        phys_conf_kwargs=None,
        paulinet_kwargs=None,
    ):
        hamil = hamil or Helpers.hamil()
        return_phys_conf = phys_conf is None
        phys_conf = phys_conf or Helpers.phys_conf(hamil, R, **(phys_conf_kwargs or {}))
        paulinet = Helpers.transform_model(PauliNet, hamil, **(paulinet_kwargs or {}))
        params, state = Helpers.init_model(
            paulinet, phys_conf, **(init_model_kwargs or {})
        )
        ret = (params, state, paulinet)
        if return_phys_conf:
            ret += (phys_conf,)
        return ret


def pytest_sessionstart(session):
    jax.config.update('jax_platform_name', 'cpu')
