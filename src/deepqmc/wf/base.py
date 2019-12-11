from torch import nn

from ..utils import Debuggable


class BaseWFNet(nn.Module, Debuggable):
    def __init__(self, mol):
        super().__init__()
        self.mol = mol
        n_elec = int(mol.charges.sum() - mol.charge)
        self.n_up = (n_elec + mol.spin) // 2
        self.n_down = (n_elec - mol.spin) // 2

    def tracked_parameters(self):
        return ()

    @property
    def spin_slices(self):
        return slice(None, self.n_up), slice(self.n_up, None)
