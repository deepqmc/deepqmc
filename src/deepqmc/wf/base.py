from torch import nn

from ..geom import Geomable
from ..utils import Debuggable


class BaseWFNet(nn.Module, Geomable, Debuggable):
    def tracked_parameters(self):
        return ()

    @property
    def spin_slices(self):
        return slice(None, self.n_up), slice(self.n_up, None)
