class ElectronicAsymptotic:
    r"""Calculate a multiplicative factor, that ensures correct electronic cusps.

    Computes the factor:
    :math:`-\frac{\text{cusp}}{\sum_{i<j}\alpha * (1 + \alpha r_{ij})}`, where
    :math:`r_{ij}` are the electron-electron distances.

    Args:
        cusp (float): the cusp parameter in the above equation.
        alpha (float): default 1, the :math:`\alpha` parameter in the above equation.
    """

    def __init__(self, *, cusp, alpha=1.0):
        super().__init__()
        self.cusp = cusp
        self.alpha = alpha

    def __call__(self, dists):
        return -(self.cusp / (self.alpha * (1 + self.alpha * dists))).sum(axis=-1)

    def extra_repr(self):
        return f'cusp={self.cusp}, alpha={self.alpha}'
