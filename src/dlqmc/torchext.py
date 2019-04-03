import torch


class BDet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, As):
        n = As.shape[-1]
        lus, pivots = As.btrifact()
        idx = torch.arange(1, n + 1, dtype=torch.int32).to(pivots.device)
        changed_sign = (pivots != idx).sum(dim=-1) % 2 == 1
        udets = lus.diagonal(dim1=-2, dim2=-1).prod(dim=-1)
        dets = torch.where(changed_sign, -udets, udets)
        ctx.save_for_backward(dets, As)
        return dets

    # TODO doesn't work for evaluation of Laplacian, see
    # https://github.com/pytorch/pytorch/issues/18619
    # @staticmethod
    # def backward(ctx, grad_output):
    #     dets, As = ctx.saved_tensors
    #     return (grad_output * dets)[:, None, None] * As.inverse().transpose(-1, -2)


bdet = BDet.apply
