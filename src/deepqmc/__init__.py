from .hamil import MolecularHamiltonian
from .molecule import Molecule
from .train import train

__all__ = ['MolecularHamiltonian', 'Molecule', 'train']

if __name__ == '__main__':
    from functools import partial

    import haiku as hk
    import jax
    import jax.numpy as jnp
    import kfac_jax
    from tqdm.auto import tqdm

    from deepqmc.jax.fit import fit_wf
    from deepqmc.jax.hamil.qho import QHOHamiltonian
    from deepqmc.jax.sampling import DecorrSampler, MetropolisSampler, chain
    from deepqmc.jax.wf.qho import QHOAnsatz

    hamil = QHOHamiltonian(3, 1.0, 1.0)

    @hk.without_apply_rng
    @hk.transform_with_state
    def ansatz(r):
        return QHOAnsatz(hamil)(r)

    opt = partial(
        kfac_jax.Optimizer,
        l2_reg=0,
        learning_rate_schedule=lambda k: 0.1 / (1 + k / 100),
        norm_constraint=1e-3,
        inverse_update_period=1,
        min_damping=1e-4,
        num_burnin_steps=0,
        estimation_mode='fisher_exact',
    )
    sampler = chain(DecorrSampler(20), MetropolisSampler(hamil, tau=1.0))
    steps = tqdm(range(10000))
    for _, _, E_loc, _ in fit_wf(
        jax.random.PRNGKey(0), hamil, ansatz, opt, sampler, 1000, steps, clip_width=2
    ):
        steps.set_postfix(E=f'{float(jnp.mean(E_loc)):.8f}')
