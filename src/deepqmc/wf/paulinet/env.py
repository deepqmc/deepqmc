import torch
from torch import nn

from deepqmc.torchext import fp_tensor


class EEShell(nn.Module):
    def __init__(self, coeffs, zetas):
        super().__init__()
        self.register_parameter('zetas', torch.nn.Parameter(zetas))
        self.register_parameter('coeffs', torch.nn.Parameter(coeffs))

    def __len__(self):
        return 1

    @property
    def l(self):
        return 0

    def extra_repr(self):
        return f'n_primitive={len(self.zetas)}'

    def forward(self, rs):
        rs = rs[..., 3].sqrt()
        exps = torch.exp(-self.zetas.abs() * rs[:, None])
        radials = (self.coeffs * exps).sum(dim=-1)[:, None]
        return radials


class EEBasis(nn.Module):
    def __init__(self, centers, shells):
        super().__init__()
        self.register_buffer('centers', centers)
        self.center_idxs, shells = zip(*shells)
        self.shells = nn.ModuleList(shells)
        self.s_center_idxs = torch.tensor([idx for idx, sh in self.items()])

    def __len__(self):
        return sum(map(len, self.shells))

    def items(self):
        return zip(self.center_idxs, self.shells)

    @classmethod
    def from_charges(cls, mol):
        centers = fp_tensor(mol.coords)
        shells = []
        for i, z in enumerate(mol.charges):
            n_shells = int(torch.div(z + 1, 2, rounding_mode='trunc'))
            for i in range(n_shells):
                shells.append((i, EEShell(fp_tensor([[1]]), fp_tensor([z / (i + 1)]))))
        return cls(centers, shells)

    def forward(self, diffs):
        shells = [sh(diffs[:, idx]) for idx, sh in self.items()]
        return torch.cat(shells, dim=-1)
