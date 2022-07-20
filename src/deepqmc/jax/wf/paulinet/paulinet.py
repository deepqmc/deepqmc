import haiku as hk
import jax
import jax.numpy as jnp

from deepqmc.jax.utils import pairwise_diffs
from deepqmc.jax.wf.base import WaveFunction

from ...types import Psi


def eval_log_slater(xs):
    if xs.shape[-1] == 0:
        return xs.new_ones(xs.shape[:-2]), xs.new_zeros(xs.shape[:-2])
    return jnp.linalg.slogdet(xs)


class PauliNet(WaveFunction):
    def __init__(self, mol, basis, confs=None, n_orbitals=None):
        super().__init__(mol)
        n_up, n_down = self.n_up, self.n_down
        n_orbitals = n_orbitals or max(n_up, n_down)
        self.basis = basis
        self.mo_coeff = hk.Linear(
            n_orbitals, with_bias=False, w_init=None, b_init=None, name='mo_coeff'
        )
        confs = [list(range(n_up)) + list(range(n_down))] if confs is None else confs
        self.confs = jnp.array(confs)
        self.conf_coeff = hk.Linear(
            len(confs),
            with_bias=False,
            w_init=None,
            b_init=None,
            name='conf_coeff',
        )

    def __call__(self, rs):
        n_elec = rs.shape[-2]
        diffs_nuc = pairwise_diffs(rs.reshape(-1, 3), self.mol.coords)
        aos = self.basis(diffs_nuc)
        xs = self.mo_coeff(aos)
        xs = xs.reshape(-1, 1, n_elec, xs.shape[-1])
        n_up = self.n_up
        conf_up, conf_down = self.confs[..., :n_up], self.confs[..., n_up:]
        det_up = xs[..., :n_up, conf_up].swapaxes(-2, -3)
        det_down = xs[..., n_up:, conf_down].swapaxes(-2, -3)
        sign_up, det_up = eval_log_slater(det_up)
        sign_down, det_down = eval_log_slater(det_down)
        xs = det_up + det_down
        xs_shift = xs.max(axis=(-2, -1))
        # the exp-normalize trick, to avoid over/underflow of the exponential
        xs_shift = jnp.where(~jnp.isinf(xs_shift), xs_shift, jnp.zeros_like(xs_shift))
        # replace -inf shifts, to avoid running into nans (see sloglindet)
        xs = sign_up * sign_down * jnp.exp(xs - xs_shift[..., None, None])
        psi = self.conf_coeff(xs).squeeze(axis=-1).mean(axis=-1)
        log_psi = (jnp.log(jnp.abs(psi)) + xs_shift).squeeze()
        sign_psi = (jax.lax.stop_gradient(jnp.sign(psi))).squeeze()
        psi = Psi(sign_psi, log_psi)
        return psi
