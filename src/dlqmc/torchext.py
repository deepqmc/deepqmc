import torch


def bdet(As):
    n = As.shape[-1]
    lus, pivots, info = As.btrifact_with_info()
    assert not info.nonzero().numel()
    idx = torch.arange(1, n + 1, dtype=torch.int32)
    changed_sign = (pivots != idx).sum(dim=-1) % 2 == 1
    dets = lus.diagonal(dim1=-2, dim2=-1).prod(dim=-1)
    dets = torch.where(changed_sign, -dets, dets)
    return dets
