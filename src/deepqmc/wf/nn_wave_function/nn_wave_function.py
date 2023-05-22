import haiku as hk
import jax
import jax.numpy as jnp

from ...physics import pairwise_diffs, pairwise_self_distance
from ...types import Psi
from ...utils import flatten, triu_flat
from ..base import WaveFunction
from .cusp import ElectronicAsymptotic

__all__ = ['NeuralNetworkWaveFunction']


class BackflowOp(hk.Module):
    def __init__(self, mult_act=None, add_act=None, with_envelope=True):
        super().__init__()
        self.mult_act = mult_act or (lambda x: 1 + 2 * jnp.tanh(x / 4))
        self.add_act = add_act or (lambda x: 0.1 * jnp.tanh(x / 4))
        self.with_envelope = with_envelope

    def __call__(self, xs, fs_mult, fs_add, dists_nuc):
        if fs_add is not None:
            if self.with_envelope:
                envel = jnp.sqrt((xs**2).sum(axis=-1, keepdims=True))
            else:
                envel = 1
        if fs_mult is not None:
            xs = xs * self.mult_act(fs_mult)
        if fs_add is not None:
            R = dists_nuc.min(axis=-1) / 0.5
            cutoff = jnp.where(
                R < 1, R**2 * (6 - 8 * R + 3 * R**2), jnp.ones_like(R)
            )
            idx = (slice(None), *([None] * (len(xs.shape) - 3)), slice(None), None)
            xs = xs + cutoff[idx] * envel * self.add_act(fs_add)
        return xs


def eval_log_slater(xs):
    if xs.shape[-1] == 0:
        return jnp.ones(xs.shape[:-2]), jnp.zeros(xs.shape[:-2])
    return jnp.linalg.slogdet(xs)


class NeuralNetworkWaveFunction(WaveFunction):
    r"""
    Implements the neural network wave function.

    Configuration files to obtain the PauliNet [HermannNC20]_, FermiNet [PfauPRR20]_,
    and DeepErwin [Gerard22]_ architectures are provided.

    Args:
        hamil (~MolecularHamiltonian): the Hamiltonian of the system.
        omni_factory (Callable): creates the omni net.
        envelope (~wf.nn_wave_function.env.ExponentialEnvelopes): the orbital envelopes.
        backflow_op (Callable): specifies how the backflow is applied to the
            orbitals.
        n_determinants (int): specifies the number of determinants
        full_determinant (bool): if :data:`False`, the determinants are
            factorized into spin-up and spin-down parts.
        cusp_electrons (bool): whether to apply the electronic cusp correction.
        cusp_alpha (float): the :math:`\alpha` factor of the electronic cusp
            correction.
        backflow_transform (str): describes the backflow transformation.
            Possible values:

            - ``'mult'``: the backflow is a multiplicative factor
            - ``'add'``: the backflow is an additive term
            - ``'both'``: the backflow consist of a multiplicative factor
                and an additive term
        conf_coeff (bool): whether to use trainable determinant coefficients.
    """

    def __init__(
        self,
        hamil,
        *,
        omni_factory,
        envelope,
        backflow_op,
        n_determinants,
        full_determinant,
        cusp_electrons,
        cusp_alpha,
        backflow_transform,
        conf_coeff,
    ):
        super().__init__(hamil.mol)
        n_up, n_down = self.n_up, self.n_down
        self.n_det = n_determinants
        self.full_determinant = full_determinant
        self.envelope = envelope(hamil.mol, n_determinants)
        self.conf_coeff = (
            hk.Linear(1, with_bias=False, w_init=jnp.ones, name='conf_coeff')
            if conf_coeff
            else conf_coeff
        )
        self.cusp_same, self.cusp_anti = (
            (ElectronicAsymptotic(cusp=cusp, alpha=cusp_alpha) for cusp in (0.25, 0.5))
            if cusp_electrons
            else (None, None)
        )
        backflow_spec = [
            *((n_up + n_down, n_up + n_down) if full_determinant else (n_up, n_down)),
            n_determinants,
            2 if backflow_transform == 'both' else 1,
        ]
        self.backflow_transform = backflow_transform
        self.backflow_op = backflow_op()
        self.omni = omni_factory(hamil.mol, *backflow_spec)

    def _backflow_op(self, xs, fs, dists_nuc):
        if self.backflow_transform == 'mult':
            fs_mult, fs_add = fs, None
        elif self.backflow_transform == 'add':
            fs_mult, fs_add = None, fs
        elif self.backflow_transform == 'both':
            fs_mult, fs_add = jnp.split(fs, 2, axis=-3)

        fs_add = fs_add.squeeze(axis=-3) if fs_add is not None else fs_add
        fs_mult = fs_mult.squeeze(axis=-3) if fs_mult is not None else fs_mult

        return self.backflow_op(xs, fs_mult, fs_add, dists_nuc)

    def __call__(self, phys_conf, return_mos=False):
        diffs_nuc = pairwise_diffs(phys_conf.r, phys_conf.R)
        dists_nuc = jnp.sqrt(diffs_nuc[..., -1])
        dists_elec = pairwise_self_distance(phys_conf.r, full=True)
        orb = self.envelope(diffs_nuc)
        jastrow, fs = self.omni(phys_conf) if self.omni else (None, None)
        orb_up, orb_down = (
            (orb, orb)
            if self.full_determinant
            else jnp.split(orb, (self.n_up * self.n_det,), axis=-1)
        )
        orb_up = orb_up[: self.n_up]
        orb_down = orb_down[self.n_up :]
        if fs is not None:
            orb_up = self._backflow_op(orb_up, fs[0], dists_nuc[: self.n_up])
            orb_down = self._backflow_op(orb_down, fs[1], dists_nuc[self.n_up :])
        orb_up = orb_up.reshape(self.n_up, self.n_det, -1).swapaxes(-2, -3)
        orb_down = orb_down.reshape(self.n_down, self.n_det, -1).swapaxes(-2, -3)
        if return_mos:
            return orb_up, orb_down

        if self.full_determinant:
            sign, xs = eval_log_slater(jnp.concatenate([orb_up, orb_down], axis=-2))
        else:
            sign_up, det_up = eval_log_slater(orb_up)
            sign_down, det_down = eval_log_slater(orb_down)
            sign, xs = sign_up * sign_down, det_up + det_down

        xs_shift = xs.max()
        # the exp-normalize trick, to avoid over/underflow of the exponential
        xs_shift = jnp.where(~jnp.isinf(xs_shift), xs_shift, jnp.zeros_like(xs_shift))
        # replace -inf shifts, to avoid running into nans (see sloglindet)
        xs = sign * jnp.exp(xs - xs_shift)
        psi = self.conf_coeff(xs).squeeze() if self.conf_coeff else xs.sum()
        log_psi = jnp.log(jnp.abs(psi)) + xs_shift
        sign_psi = jax.lax.stop_gradient(jnp.sign(psi))
        if self.cusp_same:
            cusp_same = self.cusp_same(
                jnp.concatenate(
                    [triu_flat(dists_elec[idxs, idxs]) for idxs in self.spin_slices],
                    axis=-1,
                )
            )
            cusp_anti = self.cusp_anti(flatten(dists_elec[: self.n_up, self.n_up :]))
            log_psi = log_psi + cusp_same + cusp_anti
        if jastrow is not None:
            log_psi = log_psi + jastrow

        return Psi(sign_psi, log_psi)
