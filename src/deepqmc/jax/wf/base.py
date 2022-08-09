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
        self.n_up, self.n_down = mol.n_up, mol.n_down

    @property
    def spin_slices(self):
        return slice(None, self.n_up), slice(self.n_up, None)

    def forward(self, rs):
        return NotImplemented
