import haiku as hk


class WaveFunction(hk.Module):
    r"""Base class for all trial wave functions.
    Shape:
        - Input, :math:`\mathbf r`, a.u.: :math:`(\cdot,N,3)`
        - Output1, :math:`\ln|\psi(\mathbf r)|`: :math:`(\cdot)`
        - Output2, :math:`\operatorname{sgn}\psi(\mathbf r)`: :math:`(\cdot)`
    """

    def __init__(self, mol):
        super().__init__()
        self.mol = mol
        n_elec = int(mol.charges.sum() - mol.charge)
        self.n_up = (n_elec + mol.spin) // 2
        self.n_down = (n_elec - mol.spin) // 2

    @property
    def spin_slices(self):
        return slice(None, self.n_up), slice(self.n_up, None)

    def forward(self, rs):
        return NotImplemented
