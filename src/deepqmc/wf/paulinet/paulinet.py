from functools import partial

import numpy as np
import torch
from torch import nn

from deepqmc import Molecule
from deepqmc.physics import pairwise_diffs, pairwise_distance
from deepqmc.torchext import triu_flat
from deepqmc.utils import NULL_DEBUG
from deepqmc.wf import WaveFunction

from .cusp import CuspCorrection, ElectronicAsymptotic
from .distbasis import DistanceBasis
from .gto import GTOBasis
from .molorb import MolecularOrbital
from .omni import OmniSchNet

if torch.__version__ >= '1.2.0':
    from torch import det
else:
    from ..torchext import bdet as det

__version__ = '0.1.0'
__all__ = ['PauliNet']


def eval_slater(xs):
    if xs.shape[-1] == 0:
        return xs.new_ones(xs.shape[:-2])
    return det(xs.contiguous())


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
          \sum_p c_p
          \det[\tilde\varphi_{\boldsymbol\theta,{\mu_p}i}^\uparrow(\mathbf r)]
          \det[\tilde\varphi_{\boldsymbol\theta,{\mu_p}i}^\downarrow(\mathbf r)]

    Here, :math:`c_p,\mu_p` define the multideterminant expansion,
    :math:`\tilde\varphi_{\boldsymbol\theta,{\mu_p}i}(\mathbf r)` are the
    backflow-transformed molecular orbitals (equivariant with respect to the
    exchange of same-spin electrons), :math:`J_{\boldsymbol\theta}(\mathbf r)`
    is the many-body Jastrow factor and :math:`\gamma` enforces correct
    electronic cusp conditions.

    The PauliNet ansatz is implemented in logspace to avoid numerical instabilities
    introduced by wave functions having meaningful values among various orders of
    magnitude. :class:`~deepqmc.wf.paulinet` returns a tuple of the form
    :math:`\big(ln|\psi|,\text{sign}(\psi)\big)`.

    Args:
        mol (:class:`~deepqmc.Molecule`): molecule whose wave function is represented
        basis (:class:`~deepqmc.wf.paulinet.GTOBasis`): basis for the molecular orbitals
        cusp_correction (bool): whether nuclear cusp correction is used
        cusp_electrons (bool): whether electronic cusp function is used
        dist_feat_dim (int): :math:`\dim(\mathbf e)`, number of distance features
        dist_feat_cutoff (float, a.u.): distance at which distance features
            go to zero
        jastrow_factory (callable): constructor for a Jastrow factor,
            :math:`(M,\dim(\mathbf e),N^\uparrow,N^\downarrow)`
            :math:`\rightarrow(\mathbf e_{ij},\mathbf e_{iI})\rightarrow J`
        backflow_factory (callable): constructor for a backflow,
            :math:`(M,\dim(\mathbf e),N^\uparrow,N^\downarrow,N_\text{orb})`
            :math:`\rightarrow(\varphi_\mu(\mathbf r_i),\mathbf e_{ij},\mathbf e_{iI})`
            :math:`\rightarrow\tilde\varphi_{\mu i}(\mathbf r)`
        r_backflow_factory (callable): constructor for a real-space backflow
        omni_factory (callable): constructor for a combined Jastrow factor and backflow,
            with interface identical to :class:`~deepqmc.wf.paulinet.OmniSchNet`
        configuration (:class:`~torch.Tensor`:math:`(N_\text{det},N)`): :math:`\mu_p`,
            orbital indexes of multireference configurations
        mo_factory (callable): passed to :class:`~deepqmc.wf.paulinet.MolecularOrbital`
            as ``net_factory``

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
        r_backflow_factory=None,
        omni_factory=None,
        configurations=None,
        mo_factory=None,
        return_log=True,
        *,
        cusp_correction=False,
        cusp_electrons=False,
        dist_feat_dim=32,
        dist_feat_cutoff=10.0,
        backflow_type='orbital',
        backflow_channels=1,
        rc_scaling=1.0,
        cusp_alpha=10.0,
    ):
        super().__init__(mol)
        n_up, n_down = self.n_up, self.n_down
        self.dist_basis = (
            DistanceBasis(dist_feat_dim, cutoff=dist_feat_cutoff, envelope='nocusp')
            if mo_factory or jastrow_factory or backflow_factory or omni_factory
            else None
        )
        if configurations is not None:
            assert configurations.shape[1] == n_up + n_down
            n_orbitals = configurations.max().item() + 1
            self.confs = configurations
            self.conf_coeff = nn.Linear(len(configurations), 1, bias=False)
        else:
            n_orbitals = max(n_up, n_down)
            self.confs = torch.cat(
                [torch.arange(n_up), torch.arange(n_down)]
            ).unsqueeze(dim=0)
            self.conf_coeff = nn.Identity()
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
            'orbital': (n_orbitals, backflow_channels),
            'det': (max(n_up, n_down), len(self.confs) * backflow_channels),
        }[backflow_type]
        self.backflow_type = backflow_type
        self.backflow = (
            backflow_factory(len(mol), dist_feat_dim, n_up, n_down, *backflow_spec)
            if backflow_factory
            else None
        )
        self.r_backflow = None
        if omni_factory:
            assert not backflow_factory and not jastrow_factory
            self.omni = omni_factory(mol, dist_feat_dim, n_up, n_down, *backflow_spec)
            self.backflow = self.omni.forward_backflow
            self.r_backflow = self.omni.forward_r_backflow
            self.jastrow = self.omni.forward_jastrow
        else:
            self.omni = None
        self.return_log = return_log

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
            kwargs: all other arguments are passed to the :class:`PauliNet`
                constructor

        .. _PySCF: http://pyscf.org
        """
        n_up, n_down = mf.mol.nelec
        if hasattr(mf, 'fcisolver'):
            if conf_limit:
                conf_cutoff = max(
                    np.sort(abs(mf.ci.flatten()))[-conf_limit] - 1e-10, conf_cutoff
                )
            conf_coeff, *confs = zip(
                *mf.fcisolver.large_ci(
                    mf.ci, mf.ncas, mf.nelecas, tol=conf_cutoff, return_strs=False
                )
            )
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
        else:
            confs = None
        mol = Molecule(
            mf.mol.atom_coords().astype('float32'),
            mf.mol.atom_charges(),
            mf.mol.charge,
            mf.mol.spin,
        )
        basis = GTOBasis.from_pyscf(mf.mol)
        wf = cls(mol, basis, configurations=confs, **kwargs)
        if init_weights:
            wf.mo.init_from_pyscf(mf, freeze_mos=freeze_mos)
            if confs is not None:
                wf.conf_coeff.weight.detach().copy_(conf_coeff)
                if freeze_confs:
                    wf.conf_coeff.weight.requires_grad_(False)
        return wf

    @classmethod
    def from_hf(
        cls, mol, *, basis='6-311g', cas=None, pauli_kwargs=None, omni_kwargs=None
    ):
        r"""Construct a :class:`PauliNet` instance by running a HF calculation.

        This is the top-level interface.

        Args:
            mol (:class:`~deepqmc.Molecule`): molecule whose wave function
                is represented
            basis (str): basis of the internal HF calculation
            cas ((int, int)): tuple of the number of active orbitals and number of
                active electrons for a complete active space multireference
                HF calculation
            pauli_kwargs: arguments passed to :func:`PauliNet.from_pyscf`
            omni_kwargs: arguments passed to :class:`~deepqmc.wf.paulinet.OmniSchNet`
        """
        from pyscf import gto, mcscf, scf

        mol = gto.M(
            atom=mol.as_pyscf(),
            unit='bohr',
            basis=basis,
            charge=mol.charge,
            spin=mol.spin,
            cart=True,
        )
        mf = scf.RHF(mol)
        mf.kernel()
        if cas:
            mc = mcscf.CASSCF(mf, *cas)
            mc.kernel()
        wf = PauliNet.from_pyscf(
            mc if cas else mf,
            **{
                'omni_factory': partial(OmniSchNet, **(omni_kwargs or {})),
                'cusp_correction': True,
                'cusp_electrons': True,
                **(pauli_kwargs or {}),
            },
        )
        wf.mf = mf
        return wf

    def forward(self, rs, debug=NULL_DEBUG):
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
            rs_flowed = self.r_backflow(rs, *edges, debug=debug)
            diffs_nuc = pairwise_diffs(
                torch.cat([coords, rs_flowed.flatten(end_dim=1)]), coords
            )
        with debug.cd('mos'):
            xs = self.mo(diffs_nuc, edges_nuc, debug=debug)
        # get orbitals as [bs, 1, i, mu]
        xs = debug['slaters'] = xs.view(batch_dim, 1, n_elec, -1)
        if self.backflow:
            with debug.cd('backflow'):
                fs = self.backflow(*edges, debug=debug)  # [bs, q, i, mu/nu]
            if self.backflow_type == 'orbital':
                xs = xs * fs
        # form dets as [bs, q, p, i, nu]
        conf_up, conf_down = self.confs[:, : self.n_up], self.confs[:, self.n_up :]
        det_up = xs[:, :, : self.n_up, conf_up].transpose(-3, -2)
        det_down = xs[:, :, self.n_up :, conf_down].transpose(-3, -2)
        if self.backflow and self.backflow_type == 'det':
            n_conf = len(self.confs)
            fs = fs.unflatten(1, ((None, fs.shape[1] // n_conf), (None, n_conf)))
            det_up = det_up * fs[..., : self.n_up, : self.n_up]
            det_down = det_down * fs[..., self.n_up :, : self.n_down]
            # with open-shell systems, part of the backflow output is not used
        if self.return_log:
            sign_up, det_up = eval_log_slater(det_up)
            sign_down, det_down = eval_log_slater(det_down)
            xs = det_up + det_down
            xs_shift = xs.flatten(start_dim=1).max(dim=-1).values
            # the exp-normalize trick, to avoid over/underflow of the exponential
            xs = sign_up * sign_down * torch.exp(xs - xs_shift[:, None, None])
        else:
            det_up = debug['det_up'] = eval_slater(det_up)
            det_down = debug['det_down'] = eval_slater(det_down)
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
            with debug.cd('jastrow'):
                psi = (
                    psi + self.jastrow(*edges, debug=debug)
                    if self.return_log
                    else psi * torch.exp(self.jastrow(*edges, debug=debug))
                )
        if self.omni:
            self.omni.forward_close()
        return (psi, sign) if self.return_log else psi
