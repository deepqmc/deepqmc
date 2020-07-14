import torch
import torch.nn as nn
from deepqmc.torchext import triu_flat


def heavyside(x):
    res = torch.zeros_like(x)
    res[x > 0] = 1
    return res


def expand_powers(r, s, e):
    return (
        torch.stack([r ** i for i in range(s, e + 1)])
        .flatten(start_dim=1)
        .permute(1, 0)
        .reshape((*r.shape, -1))
    )


def convert_idx(idx, n):
    (i, j, k) = idx
    return round((j + i * (i + 1) / 2 + k * (n * (n + 1) / 2)))


def give_inds(N_f_eN, N_f_ee):
    return (
        (i, j, k)
        for k in range(N_f_ee + 1)
        for i in range(N_f_eN + 1)
        for j in range(i + 1)
    )


def get_beta(parameters, mol, N_chi):
    beta = torch.zeros(len(mol.charges), N_chi, device='cuda')
    for i, c in enumerate(torch.unique(mol.charges)):
        beta[mol.charges == c] = parameters[i, :]
    return beta


def chi(r_en, beta_params, L_chi, N_chi, C, mol):
    Z = mol.charges.cuda()
    beta = get_beta(beta_params, mol, N_chi)
    env = (r_en - L_chi[None, :, None]) ** C * heavyside(L_chi[None, :, None] - r_en)

    return (
        (
            env
            * (
                (
                    beta[:, 0][None, :, None]
                    + (-Z / ((-L_chi) ** C) + beta[:, 0] * C / L_chi)[None, :, None]
                    * r_en
                )
                + torch.einsum(
                    'jl,ijkl->ijk', beta[:, 1:], expand_powers(r_en, 2, N_chi)
                )
            )
        )
        .sum(dim=-1)
        .mean(dim=-1)
    )


def u(r_ee, alpha_params, L_u, N_u, C, n_up, n_down):
    n = n_up + n_down
    alpha = alpha_params
    G = torch.tensor(
        [
            1 / 4
            if (
                ((i + 1) <= n_up and (j + 1) <= n_up)
                or ((i + 1) > n_up and (j + 1) > n_up)
            )
            else 1 / 2
            for i in range(n)
            for j in range(n)
        ],
        device='cuda',
    ).view(n, n)
    env = (r_ee - L_u) ** C * heavyside(L_u - r_ee)
    u_ij = env * (
        alpha[0]
        + (G / (-L_u) ** C + alpha[0] * C / L_u) * r_ee
        + torch.einsum('l,ijkl', alpha[1:], expand_powers(r_ee, 2, N_u))
    )

    return triu_flat(u_ij).sum(dim=-1)


def get_conditions(n_ee, n_eN, C, L, mol):
    conditions = []
    for i, c in enumerate(torch.unique(mol.charges)):
        n = int((n_eN + 1) * (n_eN + 2) / 2 * (n_ee + 1))

        # condition number 1
        k_max = 2 * n_eN
        ind2 = [
            [(l, m, 1) for l in range(n_eN + 1) for m in range(l) if (l + m == k)]
            for k in range(0, k_max + 1)
        ]
        ind1 = [
            [(l, l, 1) for l in range(n_eN + 1) if (2 * l == k)]
            for k in range(0, k_max + 1)
        ]
        # condition number 2
        k_max = n_eN + n_ee
        indC = [
            [(0, 0, ki) for ki in [k] if (ki <= n_ee)]
            + [
                (l, 0, n)
                for l in range(1, n_eN + 1)
                for n in range(0, n_ee + 1)
                if (l + n == k)
            ]
            for k in range(0, k_max + 1)
        ]
        indL = [
            [(1, 0, ki) for ki in [k] if (ki <= n_ee)]
            + [
                (l, 1, n)
                for l in range(1, n_eN + 1)
                for n in range(0, n_ee + 1)
                if (l + n == k)
            ]
            for k in range(0, k_max + 1)
        ]
        # condition number 3
        ind02 = [[(0, 0, n)] for n in range(1, n_ee + 1)]
        # condition number 4
        ind01 = [[(l, 0, 0)] for l in range(1, n_eN + 1)]

        A = []
        for i in range(2 * n_eN + 1):
            a = torch.zeros(n, device='cuda')
            a[list(map(lambda x: convidx(x, n_eN + 1), ind2[i]))] = 2
            a[list(map(lambda x: convidx(x, n_eN + 1), ind1[i]))] = 1
            if not ind2[i] == ind1[i] == []:
                A.append(a)

        for i in range(n_eN + n_ee + 1):
            a = torch.zeros(n, device='cuda')
            a[list(map(lambda x: convidx(x, n_eN + 1), indC[i]))] = C
            a[list(map(lambda x: convidx(x, n_eN + 1), indL[i]))] = -L
            if not indC[i] == indL[i] == []:
                A.append(a)

        for i in range(n_eN):
            a = torch.zeros(n, device='cuda')
            a[list(map(lambda x: convidx(x, n_eN + 1), ind01[i]))] = 1
            if not ind01[i] == []:
                A.append(a)

        for i in range(n_ee):
            a = torch.zeros(n, device='cuda')
            a[list(map(lambda x: convidx(x, n_eN + 1), ind02[i]))] = 1
            if not ind02[i] == []:
                A.append(a)

        nc = len(A)
        while len(A) < n:
            A.append(torch.zeros(n, device='cuda'))
        U, S, V = torch.stack(A).svd()
        conditions.append(V[:, nc:])
    return conditions


def get_gammas(parameters, conditions, n_ee, n_eN, C, L, mol):

    gamma = torch.zeros(n_eN + 1, n_eN + 1, n_ee + 1, len(mol.charges), device='cuda')
    for i, c in enumerate(torch.unique(mol.charges)):
        Y = torch.einsum('ij,j', conditions[i], parameters[i])
        for m, (i, j, k) in enumerate(give_inds(n_ee, n_eN)):
            gamma[i, j, k, mol.charges == c] = Y[m]
            if not i == j:
                gamma[j, i, k, mol.charges == c] = Y[m]
    return gamma


def f(r_ee, r_en, parameters, conditions, mol, L_f, N_f_eN, N_f_ee, C):
    gamma = get_gammas(
        parameters, conditions, n_eN=N_f_eN, n_ee=N_f_ee, C=C, L=L_f, mol=mol
    )
    L_I = L_f[[(torch.unique(mol.charges) == c).nonzero().item() for c in mol.charges]]
    diag = torch.diag(torch.ones(r_en.shape[-1])).type(torch.BoolTensor)
    s = torch.einsum(
        'abcl,abdm->abcdlm', expand(r_en, 0, N_f_eN), expand(r_en, 0, N_f_eN)
    )
    s = torch.einsum(
        'lmnb,abclm,acn->abc', gamma, s[:, :, ~diag], expand(r_ee[:, ~diag], 0, N_f_ee)
    )
    pre = ((r_en - L_I[None, :, None]) ** C)[:, :, None, :] * (
        (r_en - L_I[None, :, None]) ** C
    )[:, :, :, None]
    heavy = (
        heavyside(L_I[None, :, None] - r_en)[:, :, None, :]
        * heavyside(L_I[None, :, None] - r_en)[:, :, :, None]
    )

    return (s * heavy[:, :, ~diag] * pre[:, :, ~diag]).sum(dim=(-1, -2))


class traditional_jastrow(nn.Module):
    def __init__(
        self,
        *args,
        C=3,
        N_chi=8,
        N_u=8,
        N_f_eN=0,
        N_f_ee=0,
        L_chi=None,
        L_u=torch.tensor([1.0], device='cuda'),
        L_f=None,
        mol=None,
    ):
        super().__init__()
        self.mol = mol
        self.n_nuclei, _, self.n_up, self.n_down = args
        self.atomtypes = torch.unique(self.mol.charges)
        self.C = C
        self.N_chi = N_chi
        self.N_u = N_u
        self.N_f_eN = N_f_eN
        self.N_f_ee = N_f_ee
        self.L_chi = L_chi if L_chi else torch.ones(len(self.atomtypes), device='cuda')
        self.L_u = L_u
        self.L_f = L_f if L_f else torch.ones(len(self.atomtypes), device='cuda')
        self.params_alpha = nn.Parameter(torch.zeros(self.N_u, device='cuda'))
        self.params_beta = nn.Parameter(
            torch.zeros(len(self.atomtypes), self.N_chi, device='cuda')
        )

        if not self.N_f_ee + self.N_f_eN == 0:
            n = int(
                (self.N_f_eN + 1) * (self.N_f_eN + 2) / 2 * (self.N_f_ee + 1)
                - ((self.N_f_eN * 3 + 2 + self.N_f_ee) + self.N_f_ee + self.N_f_eN)
            )
            self.params_gamma = nn.Parameter(
                torch.zeros(len(self.atomtypes), n, device='cuda')
            )
            self.conditions_gamma = get_conditions(
                n_eN=self.N_f_eN, n_ee=self.N_f_ee, C=self.C, L=self.L_f, mol=self.mol
            )

    def forward(self, dists_elec, dists_nuc, debug):
        dists_nuc = dists_nuc.permute(0, 2, 1)
        chi_r = chi(
            dists_nuc, self.params_beta, self.L_chi, self.N_chi, self.C, self.mol
        )
        u_r = u(
            dists_elec,
            self.params_alpha,
            self.L_u,
            self.N_u,
            self.C,
            self.n_up,
            self.n_down,
        )
        f_r = (
            0
            if self.N_f_ee + self.N_f_eN == 0
            else f(
                dists_elec,
                dists_nuc,
                self.params_gamma,
                self.conditions_gamma,
                self.mol,
                self.L_f,
                N_f_ee=self.N_f_ee,
                N_f_eN=self.N_f_eN,
                C=self.C,
            )
        )
        return chi_r + u_r + f_r
