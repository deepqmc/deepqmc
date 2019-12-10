from torch import nn

from ..utils import Debuggable


class BaseWFNet(nn.Module, Debuggable):
    def tracked_parameters(self):
        return ()

    @property
    def spin_slices(self):
        return slice(None, self.n_up), slice(self.n_up, None)
