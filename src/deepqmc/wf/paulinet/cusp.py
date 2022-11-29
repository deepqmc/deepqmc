import haiku as hk


class ElectronicAsymptotic(hk.Module):
    def __init__(self, *, cusp, alpha=1.0):
        super().__init__()
        self.cusp = cusp
        self.alpha = alpha

    def __call__(self, dists):
        return -(self.cusp / (self.alpha * (1 + self.alpha * dists))).sum(axis=-1)

    def extra_repr(self):
        return f'cusp={self.cusp}, alpha={self.alpha}'
