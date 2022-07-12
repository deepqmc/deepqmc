if __name__ == '__main__':
    from functools import partial

    import haiku as hk
    import jax
    import jax.numpy as jnp
    import kfac_jax
    from tqdm.auto import tqdm

    from deepqmc.jax.fit import fit_wf
    from deepqmc.jax.hamil.qho import QHOHamiltonian
    from deepqmc.jax.sampling import DecorrSampler, MetropolisSampler
    from deepqmc.jax.wf.qho import QHOAnsatz

    hamil = QHOHamiltonian(3, 1.0, 1.0)
    ansatz = hk.without_apply_rng(hk.transform(lambda r: QHOAnsatz(hamil)(r)))
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
    sampler = MetropolisSampler(hamil, 1.0)
    sampler = DecorrSampler(sampler, 20)
    steps = tqdm(range(10000))
    for _, E_loc in fit_wf(
        jax.random.PRNGKey(0), hamil, ansatz, opt, sampler, 1000, steps, clip_width=2
    ):
        steps.set_postfix(E=f'{float(jnp.mean(E_loc)):.8f}')
