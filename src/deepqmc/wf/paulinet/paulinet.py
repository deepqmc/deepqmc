import logging

import numpy as np
import torch
from torch import nn

from deepqmc import Molecule
from deepqmc.physics import pairwise_diffs, pairwise_distance
from deepqmc.torchext import sloglindet, triu_flat
from deepqmc.wf import WaveFunction

from .cusp import CuspCorrection, ElectronicAsymptotic
from .distbasis import DistanceBasis
from .gto import GTOBasis
from .molorb import MolecularOrbital
from .omni import OmniSchNet, SubnetFactory
from .pyscfext import pyscf_from_mol
from .schnet import ElectronicSchNet

__version__ = '0.1.0'
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


class PauliNet(WaveFunction):
    r"""Implements the PauliNet ansatz from [Hermann19]_.

    Derived from :class:`WaveFunction`. This constructor provides a fully
    flexible low-level interface. See the alternative constructors for the
    high-level interfaces.

    The PauliNet ansatz combines a multireference Slater determinant expansion,
    Gaussian-type cusp-corrected molecular orbitals (:class:`MolecularOrbital`),
    electronic cusp Jastrow factor (:class:`ElectronicAsymptotic`) and many-body
    Jastrow factor and backflow transformation represented by neural networks
    that use featurized particle distances,
    :math:`\mathbf e(|\mathbf r-\mathbf r'|)` (:class:`DistanceBasis`), as input,

    .. math::
        \psi_{\boldsymbol\theta}(\mathbf r)
          =\mathrm e^{\gamma(\mathbf r)+J_{\boldsymbol\theta}(\mathbf r)}
          \sum_{pq} c_p
          \det[\tilde\varphi_{\boldsymbol\theta,q{\mu_p}i}^\uparrow(\mathbf r)]
          \det[\tilde\varphi_{\boldsymbol\theta,q{\mu_p}i}^\downarrow(\mathbf r)] \\
        \tilde\varphi_{\boldsymbol\theta,q\mu i}(\mathbf r)
          :=\big(1+2\tanh(\kappa_{\boldsymbol\theta,q\mu i}(\mathbf r))\big)
          \varphi_\mu(\mathbf r_i)

    Here, :math:`c_p,\mu_p` define the multideterminant expansion,
    :math:`\varphi_\mu(\mathbf r)` are the baseline
    single-electron molecular orbitals, :math:`J_{\boldsymbol\theta}(\mathbf r)`
    is the permutation-invariant deep Jastrow factor,
    :math:`\kappa_{\boldsymbol\theta,q\mu i}(\mathbf r)` is the :math:`q`-th
    channel of the permutation-equivariant deep backflow, and :math:`\gamma`
    enforces correct electronic cusp conditions.

    Args:
        mol (:class:`~deepqmc.Molecule`): molecule whose wave function is represented
        basis (:class:`~deepqmc.wf.paulinet.GTOBasis`): basis for the molecular orbitals
        jastrow_factory (callable): constructor for a Jastrow factor,
            :math:`(M,\dim(\mathbf e),N^\uparrow,N^\downarrow)`
            :math:`\rightarrow(\mathbf e_{ij},\mathbf e_{iI})\rightarrow J`
        backflow_factory (callable): constructor for a backflow,
            :math:`(M,\dim(\mathbf e),N^\uparrow,N^\downarrow,N_\text{orb},C)`
            :math:`\rightarrow(\mathbf e_{ij},\mathbf e_{iI})`
            :math:`\rightarrow\kappa_{q\mu i}`
        omni_factory (callable): constructor for a combined Jastrow factor and backflow,
            with interface identical to :class:`~deepqmc.wf.paulinet.OmniSchNet`
        n_configurations (int): number of electron configurations
        n_orbitals (int): number of distinct molecular orbitals used across all
            configurations if given, otherwise the larger of the number of spin-up
            and spin-down electrons
        mo_factory (callable): passed to :class:`~deepqmc.wf.paulinet.MolecularOrbital`
            as ``net_factory``
        cusp_correction (bool): whether nuclear cusp correction is used
        cusp_electrons (bool): whether electronic cusp function is used
        dist_feat_dim (int): :math:`\dim(\mathbf e)`, number of distance features
        dist_feat_cutoff (float, a.u.): distance at which distance features
            go to zero
        backflow_channels (int): :math:`C`, number of backflow channels
        omni_kwargs: arguments passed to :class:`~deepqmc.wf.paulinet.OmniSchNet`

    Attributes:
        jastrow: :class:`torch.nn.Module` representing the Jastrow factor
        backflow: :class:`torch.nn.Module` representing the backflow transformation
        conf_coeff: :class:`torch.nn.Linear` with no bias that represents
            the multireference coefficients :math:`c_p` via its :attr:`weight`
            variable of shape :math:`(1,N_\text{det})`
    """

    def __init__(
        self,
        mol,
        basis,
        jastrow_factory=None,
        backflow_factory=None,
        omni_factory=None,
        n_configurations=1,
        n_orbitals=None,
        mo_factory=None,
        return_log=True,
        use_sloglindet='training',
        *,
        cusp_correction=True,
        cusp_electrons=True,
        dist_feat_dim=32,
        dist_feat_cutoff=10.0,
        backflow_type='orbital',
        backflow_channels=1,
        backflow_transform='mult',
        rc_scaling=1.0,
        cusp_alpha=10.0,
        freeze_embed=False,
        omni_kwargs=None,
    ):
        assert use_sloglindet in {'never', 'training', 'always'}
        assert return_log or use_sloglindet == 'never'
        super().__init__(mol)
        n_up, n_down = self.n_up, self.n_down
        self.dist_basis = (
            DistanceBasis(dist_feat_dim, cutoff=dist_feat_cutoff, envelope='nocusp')
            if mo_factory or jastrow_factory or backflow_factory or omni_factory
            else None
        )
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
            net_factory=mo_factory,
            dist_feat_dim=dist_feat_dim,
            cusp_correction=cusp_correction,
            rc_scaling=rc_scaling,
        )
        self.cusp_same, self.cusp_anti = (
            (ElectronicAsymptotic(cusp=cusp, alpha=cusp_alpha) for cusp in (0.25, 0.5))
            if cusp_electrons
            else (None, None)
        )
        self.jastrow = (
            jastrow_factory(len(mol), dist_feat_dim, n_up, n_down)
            if jastrow_factory
            else None
        )
        backflow_spec = {
            'orbital': [n_orbitals, backflow_channels],
            'det': [max(n_up, n_down), len(self.confs) * backflow_channels],
        }[backflow_type]
        if backflow_transform == 'both':
            backflow_spec[1] *= 2
        self.backflow_type = backflow_type
        self.backflow_transform = backflow_transform
        self.backflow = (
            backflow_factory(len(mol), dist_feat_dim, n_up, n_down, *backflow_spec)
            if backflow_factory
            else None
        )
        self.r_backflow = None
        if omni_factory:
            assert not backflow_factory and not jastrow_factory
            self.omni = omni_factory(
                mol, dist_feat_dim, n_up, n_down, *backflow_spec, **(omni_kwargs or {})
            )
            self.backflow = self.omni.forward_backflow
            self.r_backflow = self.omni.forward_r_backflow
            self.jastrow = self.omni.forward_jastrow
        else:
            self.omni = None
        self.return_log = return_log
        if freeze_embed:
            self.requires_grad_embeddings_(False)
        self.n_determinants = len(self.confs) * backflow_channels
        self.n_backflows = 0 if not self.backflow else backflow_spec[1]
        if n_up <= 1 or n_down <= 1:
            self.use_sloglindet = 'never'
            log.warning(
                'Setting use_sloglindet to "never" as not implemented for n=0 and n=1.'
            )
        # TODO implement sloglindet for special cases n=0 and n=1
        else:
            self.use_sloglindet = use_sloglindet

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
        return {
            (cls.from_hf, 'kwargs'): cls.from_pyscf,
            (cls.from_pyscf, 'kwargs'): cls,
            (cls, 'omni_kwargs'): OmniSchNet,
            (OmniSchNet, 'schnet_kwargs'): ElectronicSchNet,
            (OmniSchNet, 'subnet_kwargs'): SubnetFactory,
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
        n_up, n_down = mf.mol.nelec
        if hasattr(mf, 'fcisolver'):
            if conf_limit:
                conf_cutoff = max(
                    np.sort(abs(mf.ci.flatten()))[-conf_limit:][0] - 1e-10, conf_cutoff,
                )
            for tol in [conf_cutoff, conf_cutoff + 2e-10]:
                conf_coeff, *confs = zip(
                    *mf.fcisolver.large_ci(
                        mf.ci, mf.ncas, mf.nelecas, tol=tol, return_strs=False
                    )
                )
                if not conf_limit or len(conf_coeff) <= conf_limit:
                    break
            else:
                raise AssertionError()
            # discard the last ci wave function if degenerate
            ns_dbl = n_up - mf.nelecas[0], n_down - mf.nelecas[1]
            conf_coeff = torch.tensor(conf_coeff)
            confs = [
                [
                    torch.arange(n_dbl, dtype=torch.long).expand(len(conf_coeff), -1),
                    torch.tensor(cfs, dtype=torch.long) + n_dbl,
                ]
                for n_dbl, cfs in zip(ns_dbl, confs)
            ]
            confs = [torch.cat(cfs, dim=-1) for cfs in confs]
            confs = torch.cat(confs, dim=-1)
            kwargs['n_configurations'] = len(confs)
            kwargs['n_orbitals'] = confs.max().item() + 1
        else:
            confs = None
        mol = Molecule(
            mf.mol.atom_coords().astype('float32'),
            mf.mol.atom_charges(),
            mf.mol.charge,
            mf.mol.spin,
        )
        basis = GTOBasis.from_pyscf(mf.mol)
        wf = cls(mol, basis, **{'omni_factory': OmniSchNet, **kwargs})
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
        return wf

    def pop_chargse(self):
        try:
            mf = self.mf
        except AttributeError:
            return super().pop_charges()
        return mf.pop(verbose=0)[1]

    def _backflow_op(self, xs, fs):
        if self.backflow_transform == 'mult':
            fs_mult, fs_add = fs, None
        elif self.backflow_transform == 'add':
            fs_mult, fs_add = None, fs
        elif self.backflow_transform == 'both':
            fs_mult, fs_add = fs[:, : fs.shape[1] // 2], fs[:, fs.shape[1] // 2 :]
        if fs_add is not None:
            envel = (xs ** 2).mean(dim=-1, keepdim=True).sqrt()
        if fs_mult is not None:
            xs = xs * (1 + 2 * torch.tanh(fs_mult / 4))
        if fs_add is not None:
            xs = xs + 0.1 * envel * torch.tanh(fs_add / 4)
        return xs

    def forward(self, rs):  # noqa: C901
        batch_dim, n_elec = rs.shape[:2]
        assert n_elec == self.confs.shape[1]
        n_atoms = len(self.mol)
        coords = self.mol.coords
        diffs_nuc = pairwise_diffs(torch.cat([coords, rs.flatten(end_dim=1)]), coords)
        edges_nuc = (
            self.dist_basis(diffs_nuc[:, :, 3].sqrt())
            if self.jastrow or self.mo.net
            else None
        )
        if self.r_backflow or self.backflow or self.cusp_same or self.jastrow:
            dists_elec = pairwise_distance(rs, rs)
        if self.r_backflow or self.backflow or self.jastrow:
            edges_nuc = edges_nuc[n_atoms:].view(batch_dim, n_elec, n_atoms, -1)
            edges = self.dist_basis(dists_elec), edges_nuc
        if self.r_backflow:
            rs_flowed = self.r_backflow(rs, *edges)
            diffs_nuc = pairwise_diffs(
                torch.cat([coords, rs_flowed.flatten(end_dim=1)]), coords
            )
        xs = self.mo(diffs_nuc, edges_nuc)
        # get orbitals as [bs, 1, i, mu]
        xs = xs.view(batch_dim, 1, n_elec, -1)
        if self.backflow:
            fs = self.backflow(*edges)  # [bs, q, i, mu/nu]
            if self.backflow_type == 'orbital':
                xs = self._backflow_op(xs, fs)
        # form dets as [bs, q, p, i, nu]
        conf_up, conf_down = self.confs[:, : self.n_up], self.confs[:, self.n_up :]
        det_up = xs[:, :, : self.n_up, conf_up].transpose(-3, -2)
        det_down = xs[:, :, self.n_up :, conf_down].transpose(-3, -2)
        if self.backflow and self.backflow_type == 'det':
            n_conf = len(self.confs)
            fs = fs.unflatten(1, ((None, fs.shape[1] // n_conf), (None, n_conf)))
            det_up = self._backflow_op(det_up, fs[..., : self.n_up, : self.n_up])
            det_down = self._backflow_op(det_down, fs[..., self.n_up :, : self.n_down])
            # with open-shell systems, part of the backflow output is not used
        if self.use_sloglindet == 'always' or (
            self.use_sloglindet == 'training' and not self.sampling
        ):
            bf_dim = det_up.shape[-4]
            if isinstance(self.conf_coeff, nn.Linear):
                conf_coeff = self.conf_coeff.weight[0]
                conf_coeff = conf_coeff.expand(bf_dim, -1).flatten() / np.sqrt(bf_dim)
            else:
                conf_coeff = det_up.new_ones(1)
            det_up = det_up.flatten(start_dim=-4, end_dim=-3).contiguous()
            det_down = det_down.flatten(start_dim=-4, end_dim=-3).contiguous()
            sign, psi = sloglindet(conf_coeff, det_up, det_down)
            sign = sign.detach()
        else:
            if self.return_log:
                sign_up, det_up = eval_log_slater(det_up)
                sign_down, det_down = eval_log_slater(det_down)
                xs = det_up + det_down
                xs_shift = xs.flatten(start_dim=1).max(dim=-1).values
                # the exp-normalize trick, to avoid over/underflow of the exponential
                xs = sign_up * sign_down * torch.exp(xs - xs_shift[:, None, None])
            else:
                det_up = eval_slater(det_up)
                det_down = eval_slater(det_down)
                xs = det_up * det_down
            psi = self.conf_coeff(xs).squeeze(dim=-1).mean(dim=-1)
            if self.return_log:
                psi, sign = psi.abs().log() + xs_shift, psi.sign().detach()
        if self.cusp_same:
            cusp_same = self.cusp_same(
                torch.cat(
                    [triu_flat(dists_elec[:, idxs, idxs]) for idxs in self.spin_slices],
                    dim=1,
                )
            )
            cusp_anti = self.cusp_anti(
                dists_elec[:, : self.n_up, self.n_up :].flatten(start_dim=1)
            )
            psi = (
                psi + cusp_same + cusp_anti
                if self.return_log
                else psi * torch.exp(cusp_same + cusp_anti)
            )
        if self.jastrow:
            J = self.jastrow(*edges)
            psi = psi + J if self.return_log else psi * torch.exp(J)
        if self.omni:
            self.omni.forward_close()
        return (psi, sign) if self.return_log else psi
