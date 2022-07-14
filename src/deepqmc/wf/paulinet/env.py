import torch
from torch import nn

from deepqmc.torchext import fp_tensor

__all__ = ()


class EEShell(nn.Module):
    def __init__(self, zetas):
        super().__init__()
        self.register_parameter('zetas', torch.nn.Parameter(zetas))

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
        return exps[:, None]


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
            # find number of occupied shells for atom
            max_elec = 0
            n_shells = 0
            for n in range(10):
                if z <= max_elec:
                    break
                else:
                    n_shells += 1
                    for m in range(n + 1):
                        max_elec += 2 * (2 * m + 1)
            # adding the lowest unoccupied shell might be beneficial,
            # especially for transition metals
            #  n_shells += 1
            for k in range(n_shells):
                shells.append((i, EEShell(fp_tensor([z / (k + 1)]))))
        return cls(centers, shells)

    def forward(self, diffs):
        shells = [sh(diffs[:, idx]) for idx, sh in self.items()]
        return torch.cat(shells, dim=-1)
