import torch
from torch.testing import assert_allclose

from deepqmc.utils import pow_int


def test_pow_int():
    xs = torch.randn(4, 3)
    exps = torch.tensor([(1, 2, 3), (0, 1, 2)])
    assert_allclose(pow_int(xs[:, None, :], exps), xs[:, None, :] ** exps.float())
