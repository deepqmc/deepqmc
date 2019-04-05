import torch

from .utils import InfoException


class LUFactError(InfoException):
    pass


class BDet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, As):
        n = As.shape[-1]
        is_zero = As.abs() < torch.finfo(As.dtype).eps
        is_nontriv = ~(is_zero.all(dim=1).any(dim=-1) | is_zero.all(dim=2).any(dim=-1))
        lus, pivots, is_fail = As[is_nontriv].btrifact_with_info()
        is_fail = is_fail == 1
        if is_fail.any():
            idxs = torch.arange(len(As))[is_nontriv][is_fail]
            raise LUFactError({'idxs': idxs, 'dets': list(map(torch.det, As[idxs]))})
        idx = torch.arange(1, n + 1, dtype=torch.int32).to(pivots.device)
        changed_sign = (pivots != idx).sum(dim=-1) % 2 == 1
        udets = lus.diagonal(dim1=-2, dim2=-1).prod(dim=-1)
        dets = As.new_zeros(len(As))
        dets[is_nontriv] = torch.where(changed_sign, -udets, udets)
        ctx.save_for_backward(As, is_nontriv)
        return dets

    @staticmethod
    def backward(ctx, grad_output):
        As, is_nontriv = ctx.saved_tensors
        grad_input = torch.zeros_like(As)
        As = As[is_nontriv]
        # TODO can we reuse results of forward()? See also:
        # https://github.com/pytorch/pytorch/issues/18619
        dets = BDet.apply(As)
        grad_input[is_nontriv] = (grad_output[is_nontriv] * dets)[
            :, None, None
        ] * As.inverse().transpose(-1, -2)
        return grad_input


bdet = BDet.apply
