import numpy as np
from torch import nn

from .. import torchext
from ..utils import NULL_DEBUG


def eval_slater(xs):
    if xs.shape[-1] == 0:
        return xs.new_ones(len(xs))
    return torchext.bdet(xs)


class NetPairwiseAntisymmetry(nn.Module):
    def __init__(self, net_pair):
        super().__init__()
        self.net_pair = net_pair

    def forward(self, x_i, x_j):
        return self.net_pair(x_i, x_j) - self.net_pair(x_j, x_i)


class PairAntisymmetric(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x1s, x2s, x12s, debug=NULL_DEBUG):
        left, right = debug['left'], debug['right'] = (
            self.net(x1s, x2s, x12s),
            self.net(x2s, x1s, x12s),
        )
        return debug.result(left - right)


class Odd(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, debug=NULL_DEBUG):
        left, right = debug['left'], debug['right'] = self.net(x), self.net(-x)
        return debug.result(left - right)


NetOdd = Odd


class AntisymmetricPart(nn.Module):
    def __init__(self, net, net_pair):
        super().__init__()
        self.net_pair_anti = NetPairwiseAntisymmetry(net_pair)
        self.net_odd = Odd(net)

    def forward(self, x):
        i, j = np.triu_indices(x.shape[-2], k=1)
        zs = self.net_pair_anti(x[:, j], x[:, i]).prod(dim=-2)
        return self.net_odd(zs)


class LaughlinAnsatz(nn.Module):
    def __init__(self, net_pair, net_odd):
        super().__init__()
        self.net_pair = PairAntisymmetric(net_pair)
        self.net_odd = Odd(net_odd)

    def forward(self, xs, xs_pair, debug=NULL_DEBUG):
        i, j = np.triu_indices(xs.shape[-2], k=1)
        with debug.cd('pair'):
            zs = self.net_pair(xs[:, i], xs[:, j], xs_pair[:, i, j], debug=debug)
            zs = zs.prod(dim=-2)
        with debug.cd('odd'):
            return debug.result(self.net_odd(zs, debug=debug))
