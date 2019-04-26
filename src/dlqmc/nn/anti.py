import numpy as np
import torch.nn as nn


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

    def forward(self, x1s, x2s, x12s):
        return self.net(x1s, x2s, x12s) - self.net(x2s, x1s, x12s)


class Odd(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x) - self.net(-x)


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

    def forward(self, xs, xs_pair):
        i, j = np.triu_indices(xs.shape[-2], k=1)
        zs = self.net_pair(xs[:, i], xs[:, j], xs_pair[:, i, j]).prod(dim=-2)
        return self.net_odd(zs)
