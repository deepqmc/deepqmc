from collections import namedtuple
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax import lax

__all__ = ()

Psi = namedtuple('Psi', 'sign log')


def laplacian(f):
    def lap(x):
        grad_f = jax.grad(f)
        df, grad_f_jvp = jax.linearize(grad_f, x)
        eye = jnp.eye(len(x))
        d2f = jnp.diag(jax.vmap(grad_f_jvp)(eye))
        return jnp.sum(d2f), df

    return lap


class QHOHamiltonian:
    def __init__(self, mass, nu):
        self.mass = mass
        self.nu = nu

    def local_energy(self, wf):
        def loc_ene(r):
            pot = 1 / 2 * self.mass * self.nu**2 * jnp.sum(r**2)
            lap_log, grad_log = laplacian(lambda x: wf(x).log)(r)
            kin = -1 / (2 * self.mass) * (lap_log + jnp.sum(grad_log**2))
            return kin + pot

        return loc_ene


class Ansatz(hk.Module):
    def __init__(self, hamil):
        super().__init__()
        self.width = 1 / jnp.sqrt(hamil.mass * hamil.nu / 2)
        self.kernel = hk.nets.MLP([64, 32, 1], activation=jax.nn.silu)

    def __call__(self, r):
        r = r / self.width
        x = self.kernel(r)
        x = jnp.squeeze(x, axis=-1)
        env = -jnp.sum(r**2, axis=-1)
        psi = Psi(jnp.sign(x), env + jnp.log(jnp.abs(x)))
        return psi


class MetropolisSampler:
    def __init__(self, hamil, tau):
        self.hamil = hamil
        self.tau = tau

    def _update(self, state, wf):
        state = {**state, 'psi': wf(state['r'])}
        return state

    def init(self, rng, wf, n):
        state = {'r': jax.random.normal(rng, (n, 3))}
        state = self._update(state, wf)
        return state

    def sample(self, state, rng, wf):
        rng_prop, rng_acc = jax.random.split(rng)
        r = state['r']
        prop = {'r': r + self.tau * jax.random.normal(rng_prop, r.shape)}
        prop = self._update(prop, wf)
        prob = jnp.exp(2 * (prop['psi'].log - state['psi'].log))
        accepted = prob > jax.random.uniform(rng_acc, prob.shape)
        state = jax.tree_map(
            lambda xp, x: jax.vmap(jnp.where)(accepted, xp, x), prop, state
        )
        return state['r'], state


class DecorrSampler:
    def __init__(self, sampler, decorr):
        self.sampler = sampler
        self.decorr = decorr

    def init(self, *args):
        return self.sampler.init(*args)

    def sample(self, state, rng, wf):
        for _, rng in zip(range(self.decorr), hk.PRNGSequence(rng)):
            _, state = self.sampler.sample(state, rng, wf)
        return state['r'], state


def fit_wf(rng, hamil, ansatz, opt, sampler, sample_size, steps):
    def loss_fn(params, r):
        wf = partial(ansatz.apply, params)
        E_loc = jax.vmap(hamil.local_energy(wf))(r)
        loss = (lax.stop_gradient(E_loc - E_loc.mean()) * wf(r).log).mean()
        return loss, E_loc

    @jax.jit
    def train_step(rng, params, opt_state, smpl_state):
        r, smpl_state = sampler.sample(smpl_state, rng, partial(ansatz.apply, params))
        grads, E_loc = jax.grad(loss_fn, has_aux=True)(params, r)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        ene = jnp.mean(E_loc)
        return params, opt_state, smpl_state, ene

    params = ansatz.init(rng, jnp.zeros(3))
    opt_state = opt.init(params)
    smpl_state = sampler.init(rng, partial(ansatz.apply, params), sample_size)
    train_state = params, opt_state, smpl_state
    for _, rng in zip(steps, hk.PRNGSequence(rng)):
        *train_state, ene = train_step(rng, *train_state)
        yield params, ene


if __name__ == '__main__':
    from tqdm.auto import tqdm

    hamil = QHOHamiltonian(1.0, 1.0)
    ansatz = hk.without_apply_rng(hk.transform(lambda r: Ansatz(hamil)(r)))
    opt = optax.adam(1e-2)
    sampler = MetropolisSampler(hamil, 1.0)
    sampler = DecorrSampler(sampler, 20)
    steps = tqdm(range(1000))
    for _, ene in fit_wf(
        jax.random.PRNGKey(0), hamil, ansatz, opt, sampler, 1000, steps
    ):
        steps.set_postfix(E=ene)
