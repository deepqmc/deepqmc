import haiku as hk
import jax
import jax.numpy as jnp
from functools import partial

from deepqmc.jax.utils import pairwise_diffs
from deepqmc.jax.wf.base import WaveFunction

from ...jaxext import unflatten
from ...types import Psi
from .omni import OmniNet


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
        return xs.new_ones(xs.shape[:-2]), xs.new_zeros(xs.shape[:-2])
    return jnp.linalg.slogdet(xs)


class PauliNet(WaveFunction):
    def __init__(
        self,
        mol,
        basis,
        confs=None,
        n_orbitals=None,
        backflow_op=None,
        *,
        full_determinant=False,
        backflow_type='orbital',
        backflow_channels=1,
        backflow_transform='mult',
        embedding_dim=64,
        omni_factory=None,
        omni_kwargs=None,
    ):

        assert not full_determinant or backflow_type == 'det'
        super().__init__(mol)
        n_up, n_down = self.n_up, self.n_down
        n_orbitals = n_orbitals or max(n_up, n_down)
        self.confs = (
            jnp.array([list(range(n_up)) + list(range(n_down))])
            if confs is None
            else confs
        )
        self.basis = basis
        self.mo_coeff = hk.Linear(n_orbitals, with_bias=False, name='mo_coeff')
        self.conf_coeff = hk.Linear(len(self.confs), with_bias=False, name='conf_coeff')
        self.n_determinants = len(self.confs) * backflow_channels
        self.full_determinant = full_determinant

        backflow_spec = {
            'orbital': [n_orbitals, backflow_channels],
            'det': [
                n_up + n_down if full_determinant else (n_up, n_down),
                len(self.confs) * backflow_channels,
            ],
        }[backflow_type]
        if backflow_transform == 'both':
            backflow_spec[1] *= 2
        self.backflow_type = backflow_type
        self.backflow_transform = backflow_transform
        self.backflow_op = backflow_op or BackflowOp()

        if omni_factory is None:
            omni_factory = partial(
                OmniNet,
                **(omni_kwargs or {}),
            )
        self.omni = omni_factory(mol, *backflow_spec)

    def _backflow_op(self, xs, fs, dists_nuc):
        if self.backflow_transform == 'mult':
            fs_mult, fs_add = fs, None
        elif self.backflow_transform == 'add':
            fs_mult, fs_add = None, fs
        elif self.backflow_transform == 'both':
            fs_mult, fs_add = fs[:, : fs.shape[1] // 2], fs[:, fs.shape[1] // 2 :]
        return self.backflow_op(xs, fs_mult, fs_add, dists_nuc)

    def __call__(self, rs):
        n_elec = rs.shape[-2]
        n_nuc = len(self.mol.coords)
        diffs_nuc = pairwise_diffs(rs.reshape(-1, 3), self.mol.coords)
        dists_nuc = jnp.sqrt(diffs_nuc[..., -1]).reshape(-1, n_elec, n_nuc)
        aos = self.basis(diffs_nuc)
        xs = self.mo_coeff(aos)
        xs = xs.reshape(-1, 1, n_elec, xs.shape[-1])
        J, fs = self.omni(rs)
        if fs is not None and self.backflow_type == 'orbital':
            xs = self._backflow_op(xs, fs, dists_nuc)

        n_up = self.n_up
        conf_up, conf_down = self.confs[..., :n_up], self.confs[..., n_up:]
        det_up = xs[..., :n_up, conf_up].swapaxes(-2, -3)
        det_down = xs[..., n_up:, conf_down].swapaxes(-2, -3)

        if fs is not None and self.backflow_type == 'det':
            n_conf = len(self.confs)
            fs = (
                unflatten(fs[0], 1, (fs[0].shape[-3] // n_conf, n_conf)),
                unflatten(fs[1], 1, (fs[1].shape[-3] // n_conf, n_conf)),
            )
            det_up = self._backflow_op(det_up, fs[0], dists_nuc[..., :n_up, :])
            det_down = self._backflow_op(det_down, fs[1], dists_nuc[..., n_up:, :])

        sign_up, det_up = eval_log_slater(det_up)
        sign_down, det_down = eval_log_slater(det_down)
        xs = det_up + det_down
        xs_shift = xs.max(axis=(-2, -1))
        # the exp-normalize trick, to avoid over/underflow of the exponential
        xs_shift = jnp.where(~jnp.isinf(xs_shift), xs_shift, jnp.zeros_like(xs_shift))
        # replace -inf shifts, to avoid running into nans (see sloglindet)
        xs = sign_up * sign_down * jnp.exp(xs - xs_shift[..., None, None])
        psi = self.conf_coeff(xs).squeeze(axis=-1).mean(axis=-1)
        log_psi = jnp.log(jnp.abs(psi)) + xs_shift
        sign_psi = jax.lax.stop_gradient(jnp.sign(psi))
        if J is not None:
            log_psi = log_psi + J

        return Psi(sign_psi.squeeze(), log_psi.squeeze())
