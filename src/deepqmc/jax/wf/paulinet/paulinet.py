import haiku as hk
import jax
import jax.numpy as jnp

from ...physics import pairwise_diffs, pairwise_self_distance
from ...types import Psi
from ...utils import flatten, triu_flat, unflatten
from ..base import WaveFunction
from .cusp import ElectronicAsymptotic
from .env import ExponentialEnvelopes
from .omni import OmniNet

__all__ = ['PauliNet']


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


class PauliNet(WaveFunction):
    r"""
    Implement the PauliNet Ansatz from [HermannNC20]_.

    Args:
        hamil (~jax.MolecularHamiltonian): the Hamiltonian of the system.
        basis (~jax.wf.paulinet.env.ExponentialEnvelopes): optional, the orbital
            envelopes.
        confs (int, (:math:`N_\text{conf}`, :math:`N_\text{elec}`)): optional,
            specifies the electronic configuration, i.e. the occupied orbitals
            in each determinant. The value of :data:`confs[i,j]` is the index of
            the orbital occupied by the :data:`j` th electron in the :data:`i` th
            determinant.
        backflow_op (Callable): optional, specifies how the backflow is applied
            to the orbitals.
        full_determinant (bool): optional, if :data:`False`, the determinants are
            factorized into spin-up and spin-down parts.
        cusp_electrons (bool): optional, whether to apply the electronic cusp
            correction.
        cusp_alpha (float): optional, the :math:`\alpha` factor of the electronic
            cusp correction.
        backflow_type (str): optional, specifies the type of backflow applied:

            - ``'orbital'``: the same backflow is applied to an orbital in
                    all determinants
            - ``'determinant'``: separate backflows are applied to each orbital in
                    each determinant
        backflow_channels (int): optional, the number of chanells of backflow.
        backflow_transform (str): optional, describes the backflow transformation.
            Possible values:

            - ``'mult'``: the backflow is a multiplicative factor
            - ``'add'``: the backflow is an additive term
            - ``'both'``: the backflow consist of a multiplicative factor
                and an additive term
        omni_factory (Callable): optional, creates the omni net.
        omni_kwargs (dict): optional, extra arguments passed to
            :data:`omni_factory`.
    """

    def __init__(
        self,
        hamil,
        basis=None,
        confs=None,
        backflow_op=None,
        *,
        full_determinant=False,
        cusp_electrons=True,
        cusp_alpha=10.0,
        backflow_type='orbital',
        backflow_channels=1,
        backflow_transform='mult',
        omni_factory=None,
        omni_kwargs=None,
        mo_scaling: float = 1.0,
    ):
        super().__init__(hamil.mol)
        n_up, n_down = self.n_up, self.n_down
        confs = (
            (
                [list(range(n_up)) + list(range(n_down))]
                if not full_determinant
                else [list(range(n_up + n_down)) * 2]
            )
            if confs is None
            else confs
        )
        self.n_orbitals = max(sum(confs, [])) + 1
        self.confs = jnp.array(confs)
        self.basis = basis or ExponentialEnvelopes.from_mol(hamil.mol)
        self.mo_coeff = hk.Linear(
            self.n_orbitals,
            with_bias=False,
            w_init=lambda s, d: hk.initializers.VarianceScaling(mo_scaling)(s, d)
            + jnp.ones(s),
            name='mo_coeff',
        )
        self.conf_coeff = hk.Linear(
            1, with_bias=False, w_init=jnp.ones, name='conf_coeff'
        )
        self.cusp_same, self.cusp_anti = (
            (ElectronicAsymptotic(cusp=cusp, alpha=cusp_alpha) for cusp in (0.25, 0.5))
            if cusp_electrons
            else (None, None)
        )
        self.n_determinants = len(self.confs) * backflow_channels
        self.full_determinant = full_determinant

        backflow_spec = {
            'orbital': [self.n_orbitals, backflow_channels],
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

        omni_factory = omni_factory or OmniNet
        self.omni = omni_factory(hamil.mol, *backflow_spec, **(omni_kwargs or {}))
        self.full_determinant = full_determinant

    def _backflow_op(self, xs, fs, dists_nuc):
        if self.backflow_transform == 'mult':
            fs_mult, fs_add = fs, None
        elif self.backflow_transform == 'add':
            fs_mult, fs_add = None, fs
        elif self.backflow_transform == 'both':
            fs_mult, fs_add = fs[:, : fs.shape[1] // 2], fs[:, fs.shape[1] // 2 :]
        return self.backflow_op(xs, fs_mult, fs_add, dists_nuc)

    def __call__(self, rs, return_mos=False):
        n_elec = rs.shape[-2]
        n_nuc = len(self.mol.coords)
        diffs_nuc = pairwise_diffs(rs.reshape(-1, 3), self.mol.coords)
        dists_nuc = jnp.sqrt(diffs_nuc[..., -1]).reshape(-1, n_elec, n_nuc)
        dists_elec = pairwise_self_distance(rs, full=True)
        aos = self.basis(diffs_nuc)
        xs = self.mo_coeff(aos)
        xs = jnp.expand_dims(xs, axis=-3)
        J, fs = self.omni(rs) if self.omni else (None, None)
        if fs is not None and self.backflow_type == 'orbital':
            xs = self._backflow_op(xs, fs, dists_nuc)
        n_up = self.n_up
        n_slice = n_up + self.n_down if self.full_determinant else n_up
        conf_up, conf_down = self.confs[..., :n_slice], self.confs[..., n_slice:]
        det_up = xs[..., :n_up, conf_up].swapaxes(-2, -3)
        det_down = xs[..., n_up:, conf_down].swapaxes(-2, -3)
        if self.full_determinant:
            det_full = jnp.zeros((*det_up.shape[:-2], n_elec, n_elec))
            det_full = det_full.at[..., :n_up, :].set(det_up)
            det_full = det_full.at[..., n_up:, :].set(det_down)
            det_up = det_full
            det_down = jnp.empty((*det_down.shape[:-2], 0, 0))
        if fs is not None and self.backflow_type == 'det':
            n_conf = len(self.confs)
            if self.full_determinant:
                fs = (unflatten(fs, -3, (fs.shape[-3] // n_conf, n_conf)), None)
            else:
                fs = (
                    unflatten(fs[0], -3, (fs[0].shape[-3] // n_conf, n_conf)),
                    unflatten(fs[1], -3, (fs[1].shape[-3] // n_conf, n_conf)),
                )
            det_up = self._backflow_op(det_up, fs[0], dists_nuc[..., :n_up, :])
            det_down = self._backflow_op(det_down, fs[1], dists_nuc[..., n_up:, :])
        if return_mos:
            return det_up, det_down
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
        if self.cusp_same:
            cusp_same = self.cusp_same(
                jnp.concatenate(
                    [
                        triu_flat(dists_elec[..., idxs, idxs])
                        for idxs in self.spin_slices
                    ],
                    axis=-1,
                )
            )
            cusp_anti = self.cusp_anti(
                flatten(dists_elec[..., :n_up, n_up:], start_axis=-2)
            )
            log_psi = log_psi + cusp_same + cusp_anti
        if J is not None:
            log_psi = log_psi + J

        return Psi(sign_psi.squeeze(), log_psi.squeeze())
