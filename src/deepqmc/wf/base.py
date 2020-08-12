import torch
from torch import nn


class WaveFunction(nn.Module):
    r"""Base class for all trial wave functions.

    Shape:
        - Input, :math:`\mathbf r`, a.u.: :math:`(\cdot,N,3)`
        - Output1, :math:`\ln|\psi(\mathbf r)|`: :math:`(\cdot)`
        - Output2, :math:`\operatorname{sgn}\psi(\mathbf r)`: :math:`(\cdot)`
    """

    def __init__(self, mol):
        super().__init__()
        self.sampling = True
        self.mol = mol
        n_elec = int(mol.charges.sum() - mol.charge)
        self.n_up = (n_elec + mol.spin) // 2
        self.n_down = (n_elec - mol.spin) // 2

    def tracked_parameters(self):
        return ()

    @property
    def spin_slices(self):
        return slice(None, self.n_up), slice(self.n_up, None)

    def pop_charges(self):
        return torch.zeros_like(self.mol.charges)

    def forward(self, rs):
        return NotImplemented

    def sample(self, mode=True):
        self.sampling = mode
        return self
