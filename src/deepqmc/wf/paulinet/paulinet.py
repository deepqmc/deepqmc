import logging

import numpy as np
import torch
from torch import nn

from deepqmc import Molecule
from deepqmc.physics import pairwise_diffs, pairwise_distance
from deepqmc.plugins import PLUGINS
from deepqmc.torchext import sloglindet, triu_flat
from deepqmc.wf import WaveFunction

from .cusp import CuspCorrection, ElectronicAsymptotic
from .gto import GTOBasis
from .molorb import MolecularOrbital
from .omni import OmniSchNet
from .pyscfext import pyscf_from_mol

__version__ = '0.2.0'
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
        return_log=True,
        use_sloglindet='training',
        *,
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
        assert use_sloglindet in {'never', 'training', 'always'}
        assert return_log or use_sloglindet == 'never'
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
            'det': [max(n_up, n_down), len(self.confs) * backflow_channels],
        }[backflow_type]
        if backflow_transform == 'both':
            backflow_spec[1] *= 2
        self.backflow_type = backflow_type
        self.backflow_transform = backflow_transform
        if 'paulinet.omni_factory' in PLUGINS:
            log.info('Using a plugin for paulinet.omni_factory')
            omni_factory = PLUGINS['paulinet.omni_factory']
        elif isinstance(omni_factory, str):
            if omni_kwargs:
                omni_kwargs = omni_kwargs[omni_factory]
            omni_factory = self.OMNI_FACTORIES[omni_factory]
        self.omni = (
            omni_factory(
                len(mol.coords), n_up, n_down, *backflow_spec, **(omni_kwargs or {})
            )
            if omni_factory
            else None
        )
        self.return_log = return_log
        if freeze_embed:
            self.requires_grad_embeddings_(False)
        self.n_determinants = len(self.confs) * backflow_channels
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
                    np.sort(abs(mf.ci.flatten()))[-conf_limit:][0] - 1e-10, conf_cutoff
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
        wf = cls(mol, basis, **kwargs)
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

    def pop_charges(self):
        try:
            mf = self.mf
        except AttributeError:
            return super().pop_charges()
        return self.mol.charges.new(mf.pop(verbose=0)[1])

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
        dists_elec = pairwise_distance(rs, rs)
        if self.omni:
            dists_nuc = (
                diffs_nuc[n_atoms:, :, 3].sqrt().view(batch_dim, n_elec, n_atoms)
            )
        xs = self.mo(diffs_nuc)
        # get orbitals as [bs, 1, i, mu]
        xs = xs.view(batch_dim, 1, n_elec, -1)
        # get jastrow J and backflow fs (as [bs, q, i, mu/nu])
        J, fs = self.omni(dists_nuc, dists_elec) if self.omni else (None, None)
        if fs is not None and self.backflow_type == 'orbital':
            xs = self._backflow_op(xs, fs)
        # form dets as [bs, q, p, i, nu]
        conf_up, conf_down = self.confs[:, : self.n_up], self.confs[:, self.n_up :]
        det_up = xs[:, :, : self.n_up, conf_up].transpose(-3, -2)
        det_down = xs[:, :, self.n_up :, conf_down].transpose(-3, -2)
        if fs is not None and self.backflow_type == 'det':
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
                xs_shift = xs_shift.where(
                    ~torch.isinf(xs_shift), xs_shift.new_tensor(0)
                )
                # replace -inf shifts, to avoid running into nans (see sloglindet)
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
        if J is not None:
            psi = psi + J if self.return_log else psi * torch.exp(J)
        return (psi, sign) if self.return_log else psi
