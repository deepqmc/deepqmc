import logging

import numpy as np
import torch
from torch import nn

from deepqmc import Molecule
from deepqmc.physics import pairwise_diffs, pairwise_self_distance
from deepqmc.plugins import PLUGINS
from deepqmc.torchext import sloglindet, triu_flat
from deepqmc.wf import WaveFunction

from .cusp import CuspCorrection, ElectronicAsymptotic
from .gto import GTOBasis
from .molorb import MolecularOrbital
from .omni import OmniSchNet
from .pyscfext import confs_from_mc, pyscf_from_mol

__version__ = '0.4.0'
__all__ = ['PauliNet']

log = logging.getLogger(__name__)


def eval_slater(xs):
    if xs.shape[-1] == 0:
        return xs.new_ones(xs.shape[:-2])
    return torch.det(xs.contiguous())


def eval_log_slater(xs):
    if xs.shape[-1] == 0:
        return xs.new_ones(xs.shape[:-2]), xs.new_zeros(xs.shape[:-2])
    return xs.contiguous().slogdet()


class BackflowOp(nn.Module):
    def __init__(self, mult_act=None, add_act=None, with_envelope=True):
        super().__init__()
        self.mult_act = mult_act or (lambda x: 1 + 2 * torch.tanh(x / 4))
        self.add_act = add_act or (lambda x: 0.1 * torch.tanh(x / 4))
        self.with_envelope = with_envelope

    def forward(self, xs, fs_mult, fs_add, dists_nuc):
        if fs_add is not None:
            if self.with_envelope:
                envel = (xs**2).sum(dim=-1, keepdim=True).sqrt()
            else:
                envel = 1
        if fs_mult is not None:
            xs = xs * self.mult_act(fs_mult)
        if fs_add is not None:
            R = dists_nuc.min(dim=-1).values / 0.5
            cutoff = torch.where(
                R < 1, R**2 * (6 - 8 * R + 3 * R**2), R.new_tensor(1)
            )
            idx = (slice(None), *([None] * (len(xs.shape) - 3)), slice(None), None)
            xs = xs + cutoff[idx] * envel * self.add_act(fs_add)
        return xs


class PauliNet(WaveFunction):
    r"""Implements the PauliNet ansatz from [HermannNC20]_.

    Derived from :class:`WaveFunction`. This constructor provides a fully
    flexible low-level interface. See the alternative constructors for the
    high-level interfaces.

    The PauliNet ansatz combines a multireference Slater determinant expansion,
    Gaussian-type cusp-corrected molecular orbitals (:class:`MolecularOrbital`),
    electronic cusp Jastrow factor (:class:`ElectronicAsymptotic`) and
    trainable Jastrow factor and backflow transformation (:class:`OmniSchNet`).

    .. math::
        \psi_{\boldsymbol\theta}(\mathbf r) =\mathrm e^{\gamma(\mathbf
        r)+J_{\boldsymbol\theta}(\mathbf x_i^{(L)})}
        \sum_{pq}\frac{c_p}{N_\text{bf}}
        \det[\tilde{\boldsymbol\varphi}_{\boldsymbol\theta,pq}^\uparrow(\mathbf
        r)]
        \det[\tilde{\boldsymbol\varphi}_{\boldsymbol\theta,pq}^\downarrow(\mathbf
        r)] \\ \tilde\varphi_{\boldsymbol\theta,pq\nu i}(\mathbf r)
        :=\varphi_{\mu_{p\nu}}(\mathbf r_i)\star\big(\mathbf f_{*_1}(\mathbf
        x_i^{(L)})\big)_{*_2} \\ (*_1,*_2)=\begin{cases} (q,\mu_{p\nu}) &
        \text{orbital-type backflow} \\ (pq,\nu) & \text{determinant-type
        backflow} \end{cases}

    Here, :math:`\mathbf x_i^{(L)}` are the permutation-equivariant electron
    embeddings obtained from a graph neural network, :math:`\mu_{p\nu}`
    defines the electron configurations with the corresponding linear coefficients
    :math:`c_p`, :math:`\varphi_\mu(\mathbf r)` are
    the baseline single-electron molecular orbitals,
    :math:`J_{\boldsymbol\theta}` is the permutation-invariant deep Jastrow
    factor, :math:`\mathbf f_{\boldsymbol\theta}` is the
    permutation-equivariant deep backflow, and :math:`\gamma`
    enforces correct electronic cusp conditions.
    Based on the type of the backflow, the backflow is applied either
    orbital-wise or determinant-wise, and :math:`\star` can be either multiplication
    or addition.
    :math:`N_\text{bf}` different backflows indexed by :math:`q` are used with
    each electron configuration.

    Args:
        mol (:class:`~deepqmc.Molecule`): molecule whose wave function is represented
        basis (:class:`~deepqmc.wf.paulinet.GTOBasis`): basis for the molecular orbitals
        n_configurations (int): number of electron configurations
        n_orbitals (int): number of distinct molecular orbitals used across all
            configurations if given, otherwise the larger of the number of spin-up
            and spin-down electrons
        cusp_correction (bool): whether nuclear cusp correction is used
        cusp_electrons (bool): whether electronic cusp function is used
        backflow_type (str): ``'orbital'`` for orbital-type backflow, ``'det'``
            for determinant-type backflow
        backflow_channels (int): :math:`N_\text{bf}`, number of backflow channels
        backflow_transform (str): specifies the :math:`\star` operation:

            - ``'mult'`` -- :math:`x\star y:=x\big(1+2\tanh(y)\big)`
        omni_factory (callable): constructor for combined a Jastrow and backflow,
            :math:`(M,N^\uparrow,N^\downarrow,N_\text{orb},N_\text{bf})`
            :math:`\rightarrow(r_{ij},R_{iI})\rightarrow (J,f_{q\mu i})`
        omni_kwargs: extra arguments passed to ``omni_factory``

    Attributes:
        omni: :class:`torch.nn.Module` representing Jastrow and backflow
        conf_coeff: :class:`torch.nn.Linear` with no bias that represents
            the multireference coefficients :math:`c_p` via its :attr:`weight`
            variable of shape :math:`(1,N_\text{det})`
    """

    OMNI_FACTORIES = {'omni_schnet': OmniSchNet}

    def __init__(
        self,
        mol,
        basis,
        n_configurations=1,
        n_orbitals=None,
        use_sloglindet=True,
        backflow_op=None,
        dummy_coords=None,
        *,
        full_determinant=False,
        cusp_correction=True,
        cusp_electrons=True,
        backflow_type='orbital',
        backflow_channels=1,
        backflow_transform='mult',
        rc_scaling=1.0,
        cusp_alpha=10.0,
        freeze_embed=False,
        omni_factory='omni_schnet',
        omni_kwargs=None,
    ):
        assert not full_determinant or backflow_type == 'det'
        super().__init__(mol)
        n_up, n_down = self.n_up, self.n_down
        n_orbitals = n_orbitals or max(n_up, n_down)
        confs = [list(range(n_up)) + list(range(n_down))] + [
            sum((torch.randperm(n_orbitals)[:n].tolist() for n in (n_up, n_down)), [])
            for _ in range(n_configurations - 1)
        ]
        self.register_buffer('confs', torch.tensor(confs))
        self.conf_coeff = (
            nn.Linear(n_configurations, 1, bias=False)
            if n_configurations > 1
            else nn.Identity()
        )
        self.mo = MolecularOrbital(
            mol,
            basis,
            n_orbitals,
            cusp_correction=cusp_correction,
            rc_scaling=rc_scaling,
        )
        self.cusp_same, self.cusp_anti = (
            (ElectronicAsymptotic(cusp=cusp, alpha=cusp_alpha) for cusp in (0.25, 0.5))
            if cusp_electrons
            else (None, None)
        )
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
        self.register_buffer(
            'dummy_coords', torch.tensor([] if dummy_coords is None else dummy_coords)
        )
        if 'paulinet.omni_factory' in PLUGINS:
            log.info('Using a plugin for paulinet.omni_factory')
            omni_factory = PLUGINS['paulinet.omni_factory']
        elif isinstance(omni_factory, str):
            if omni_kwargs:
                omni_kwargs = omni_kwargs[omni_factory]
            omni_factory = self.OMNI_FACTORIES[omni_factory]
        self.omni = (
            omni_factory(
                len(mol.coords) + len(self.dummy_coords),
                n_up,
                n_down,
                *backflow_spec,
                **(omni_kwargs or {}),
            )
            if omni_factory
            else None
        )
        if freeze_embed:
            self.requires_grad_embeddings_(False)
        self.n_determinants = len(self.confs) * backflow_channels
        self.use_sloglindet = use_sloglindet
        self.full_determinant = full_determinant

    def requires_grad_classes_(self, classes, requires_grad):
        for m in self.modules():
            if isinstance(m, classes):
                for p in m.parameters(recurse=False):
                    p.requires_grad_(requires_grad)
        return self

    def requires_grad_cusps_(self, requires_grad):
        return self.requires_grad_classes_(CuspCorrection, requires_grad)

    def requires_grad_embeddings_(self, requires_grad):
        return self.requires_grad_classes_(nn.Embedding, requires_grad)

    def requires_grad_nets_(self, requires_grad):
        return self.requires_grad_classes_(nn.Linear, requires_grad)

    @classmethod
    def DEFAULTS(cls):
        from .omni import Backflow, Jastrow
        from .schnet import ElectronicSchNet, SubnetFactory

        return {
            (cls.from_hf, 'kwargs'): cls.from_pyscf,
            (cls.from_pyscf, 'kwargs'): cls,
            (cls, 'omni_kwargs'): cls.OMNI_FACTORIES,
            (OmniSchNet, 'schnet_kwargs'): ElectronicSchNet,
            (OmniSchNet, 'mf_schnet_kwargs'): (ElectronicSchNet, ['version']),
            (OmniSchNet, 'subnet_kwargs'): SubnetFactory,
            (OmniSchNet, 'mf_subnet_kwargs'): SubnetFactory,
            (OmniSchNet, 'jastrow_kwargs'): Jastrow,
            (OmniSchNet, 'backflow_kwargs'): Backflow,
            (OmniSchNet, 'rs_backflow_kwargs'): None,
            (Jastrow, 'kwargs'): None,
            (Backflow, 'kwargs'): None,
            (SubnetFactory, 'kwargs'): None,
        }

    @classmethod
    def from_pyscf(
        cls,
        mf,
        *,
        init_weights=True,
        freeze_mos=True,
        freeze_confs=False,
        conf_cutoff=1e-2,
        conf_limit=None,
        conf_strs=None,
        **kwargs,
    ):
        r"""Construct a :class:`PauliNet` instance from a finished PySCF_ calculation.

        Args:
            mf (:class:`pyscf.scf.hf.RHF` | :class:`pyscf.mcscf.mc1step.CASSCF`):
                restricted (multireference) HF calculation
            init_weights (bool): whether molecular orbital coefficients and
                configuration coefficients are initialized from the HF calculation
            freeze_mos (bool): whether the MO coefficients are frozen for
                gradient optimization
            freeze_confs (bool): whether the configuration coefficients are
                frozen for gradient optimization
            conf_cutoff (float): determinants with a linear coefficient above
                this threshold are included in the determinant expansion
            conf_limit (int): if given, at maximum the given number of configurations
                with the largest linear coefficients are used in the ansatz
            kwargs: all other arguments are passed to the :class:`PauliNet`
                constructor

        .. _PySCF: http://pyscf.org
        """
        assert not (set(kwargs) & {'n_configurations', 'n_orbitals'})
        assert not conf_strs or not conf_limit
        n_up, n_down = mf.mol.nelec
        if hasattr(mf, 'fcisolver'):
            confs = confs_from_mc(mf)
            if conf_limit:
                if abs(confs[conf_limit - 1][1] - confs[conf_limit][1]) < 1e-10:
                    conf_limit -= 1
                confs = confs[:conf_limit]
            if conf_strs:
                confs = {c[0]: c for c in confs}
                confs = [confs[s] for s in conf_strs]
            if not conf_limit and not conf_strs:
                confs = [c for c in confs if abs(c[1]) >= conf_cutoff]
                assert confs
            conf_strs, conf_coeff, confs = zip(*confs)
            conf_coeff = torch.tensor(conf_coeff)
            confs = torch.tensor(np.array(confs))
            log.info(f'Will use {len(confs)} electron configurations')
            kwargs['n_configurations'] = len(confs)
            kwargs['n_orbitals'] = confs.max().item() + 1
        else:
            confs = None
            conf_strs = None
        mol = Molecule(
            mf.mol.atom_coords().astype('float32'),
            mf.mol.atom_charges(),
            mf.mol.charge,
            mf.mol.spin,
        )
        basis = GTOBasis.from_pyscf(mf.mol)
        wf = cls(mol, basis, **kwargs)
        wf.conf_strs = conf_strs
        if init_weights:
            wf.mo.init_from_pyscf(mf, freeze_mos=freeze_mos)
            if confs is not None:
                wf.confs.detach().copy_(confs)
                if len(confs) > 1:
                    wf.conf_coeff.weight.detach().copy_(conf_coeff)
                if freeze_confs:
                    wf.conf_coeff.weight.requires_grad_(False)
        return wf

    @classmethod
    def from_hf(cls, mol, *, basis='6-311g', cas=None, workdir=None, **kwargs):
        r"""Construct a :class:`PauliNet` instance by running a HF calculation.

        This is the top-level interface.

        Args:
            mol (:class:`~deepqmc.Molecule`): molecule whose wave function
                is represented
            basis (str): basis of the internal HF calculation
            cas ((int, int)): tuple of the number of active orbitals and number of
                active electrons for a complete active space multireference
                HF calculation
            workdir (str): path where PySCF calculations are cached
            kwargs: all other arguments are passed to :func:`PauliNet.from_pyscf`
        """
        mf, mc = pyscf_from_mol(mol, basis, cas, workdir)
        assert bool(cas) == bool(mc)
        wf = PauliNet.from_pyscf(mc or mf, **kwargs)
        wf.mf = mf
        wf.mol.data = mol.data
        return wf

    def pop_charges(self):
        try:
            mf = self.mf
        except AttributeError:
            return super().pop_charges()
        return self.mol.charges.new(mf.pop(verbose=0)[1])

    def _backflow_op(self, xs, fs, dists_nuc):
        if self.backflow_transform == 'mult':
            fs_mult, fs_add = fs, None
        elif self.backflow_transform == 'add':
            fs_mult, fs_add = None, fs
        elif self.backflow_transform == 'both':
            fs_mult, fs_add = fs[:, : fs.shape[1] // 2], fs[:, fs.shape[1] // 2 :]
        return self.backflow_op(xs, fs_mult, fs_add, dists_nuc)

    def forward(self, rs):  # noqa: C901
        batch_dim, n_elec = rs.shape[:2]
        assert n_elec == self.confs.shape[1]
        dists_elec = pairwise_self_distance(rs, full=True)
        # get jastrow J, backflow fs (as [bs, q, i, mu/nu]), and real-space
        # backflow ps (as [bs, i, 3])
        coords = self.mol.coords
        J, fs, ps = (
            self.omni(rs, torch.cat([self.mol.coords, self.dummy_coords], dim=0))
            if self.omni
            else (None, None, None)
        )
        if ps is not None:
            rs = rs + ps
        diffs_nuc = pairwise_diffs(torch.cat([coords, rs.flatten(end_dim=1)]), coords)
        if self.omni:
            dists_nuc = (
                diffs_nuc[len(coords) :, :, -1].sqrt().view(batch_dim, n_elec, -1)
            )
        xs = self.mo(diffs_nuc)
        # get orbitals as [bs, 1, i, mu]
        xs = xs.view(batch_dim, 1, n_elec, -1)
        if fs is not None and self.backflow_type == 'orbital':
            xs = self._backflow_op(xs, fs, dists_nuc)
        # form dets as [bs, q, p, i, nu]
        n_up = self.n_up
        conf_up, conf_down = self.confs[:, :n_up], self.confs[:, n_up:]
        det_up = xs[:, :, :n_up, conf_up].transpose(-3, -2)
        det_down = xs[:, :, n_up:, conf_down].transpose(-3, -2)
        if fs is not None and self.backflow_type == 'det':
            n_conf = len(self.confs)
            if self.full_determinant:
                fs = fs.unflatten(1, (fs.shape[1] // n_conf, n_conf))
                det_full = fs.new_zeros((*det_up.shape[:3], n_elec, n_elec))
                det_full[..., :n_up, :n_up] = det_up
                det_full[..., n_up:, n_up:] = det_down
                det_up = det_full = self._backflow_op(det_full, fs, dists_nuc)
                det_down = fs.new_empty((*det_down.shape[:3], 0, 0))
            else:
                fs = (
                    fs[0].unflatten(1, (fs[0].shape[1] // n_conf, n_conf)),
                    fs[1].unflatten(1, (fs[1].shape[1] // n_conf, n_conf)),
                )
                det_up = self._backflow_op(det_up, fs[0], dists_nuc[:, :n_up])
                det_down = self._backflow_op(det_down, fs[1], dists_nuc[:, n_up:])
        if self.use_sloglindet:
            bf_dim = det_up.shape[-4]
            if isinstance(self.conf_coeff, nn.Linear):
                conf_coeff = self.conf_coeff.weight[0]
            else:
                conf_coeff = det_up.new_ones(1)
            conf_coeff = conf_coeff.expand(bf_dim, -1).flatten() / bf_dim
            det_up = det_up.flatten(start_dim=-4, end_dim=-3).contiguous()
            det_down = det_down.flatten(start_dim=-4, end_dim=-3).contiguous()
            sign, psi = sloglindet(conf_coeff, det_up, det_down)
            sign = sign.detach()
        else:
            sign_up, det_up = eval_log_slater(det_up)
            sign_down, det_down = eval_log_slater(det_down)
            xs = det_up + det_down
            xs_shift = xs.flatten(start_dim=1).max(dim=-1).values
            # the exp-normalize trick, to avoid over/underflow of the exponential
            xs_shift = xs_shift.where(~torch.isinf(xs_shift), xs_shift.new_tensor(0))
            # replace -inf shifts, to avoid running into nans (see sloglindet)
            xs = sign_up * sign_down * torch.exp(xs - xs_shift[:, None, None])
            psi = self.conf_coeff(xs).squeeze(dim=-1).mean(dim=-1)
            psi, sign = psi.abs().log() + xs_shift, psi.sign().detach()
        if self.cusp_same:
            cusp_same = self.cusp_same(
                torch.cat(
                    [triu_flat(dists_elec[:, idxs, idxs]) for idxs in self.spin_slices],
                    dim=1,
                )
            )
            cusp_anti = self.cusp_anti(dists_elec[:, :n_up, n_up:].flatten(start_dim=1))
            psi = psi + cusp_same + cusp_anti
        if J is not None:
            psi = psi + J
        return psi, sign
