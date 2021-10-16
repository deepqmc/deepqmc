import torch

from .utils import bdiag, idx_perm

__all__ = ()


def slog(x):
    return x.sign(), x.abs().log()


def slog_fn_exp(fn, *args, dim=None, idx=None):
    if len(args) > 1:
        assert dim is None
        dim = -1
        sgn, x = zip(*args)
        sgn, x = torch.stack(sgn, dim=dim), torch.stack(x, dim=dim)
    else:
        assert dim is not None
        sgn, x = args[0]
    if isinstance(dim, int):
        dim = [dim]
    shift = x
    for d in dim:
        shift = shift.max(dim=d, keepdim=True).values
    shift = shift.where(~torch.isinf(shift), shift.new_tensor(0))
    x = (x - shift).exp()
    if sgn is not None:
        x = sgn * x
    for d in sorted(dim):
        shift = shift.squeeze(dim=d)
    x = fn(x)
    sgn, x = slog(x)
    if idx is not None:
        shift = shift[idx]
    x = x + shift
    return sgn, x


def log_gamma(s):
    idx = idx_perm(s.shape[-1], 2, s.device)[-1]
    return s[..., idx].log().sum(dim=-1)


def log_rho(s):
    if s.shape[-1] == 2:
        return s.new_zeros(*s.shape, 1)
    idx = idx_perm(s.shape[-1], 3, s.device)[-1]
    return s[..., idx].log().sum(dim=-1)


def _slogcof(A):
    # SVD seems to be hugely suboptimal on GPU, especially for small matrices,
    # so we do everything on a CPU
    U, s, V = (x.to(A.device) for x in A.cpu().svd())
    sgn_UV = U.det().sign() * V.det().sign()
    if A.shape[-1] <= 1:
        sl_C = torch.ones_like(A), torch.zeros_like(A)
    else:
        log_g = log_gamma(s)
        idx = ..., None, None
        sl_C = slog_fn_exp(
            lambda g: sgn_UV[idx] * U @ g.diag_embed() @ V.transpose(-1, -2),
            (None, log_g),
            dim=-1,
            idx=idx,
        )
    return (*sl_C, U, s, V, sgn_UV)


def _sloglindet_ref(c, A1, A2):
    Psi = (c * A1.det() * A2.det()).sum(dim=-1)
    return Psi.sign(), Psi.abs().log()


class SLogLinearDet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, A1, A2):
        assert len(c.shape) == 1
        assert len(A1.shape) >= 3
        assert len(A2.shape) >= 3
        assert A1.shape[-3] == A2.shape[-3] == c.shape[0]
        assert A1.shape[:-3] == A2.shape[:-3]
        assert A1.shape[-1] == A1.shape[-2]
        assert A2.shape[-1] == A2.shape[-2]
        sl_D1 = A1.slogdet()
        sl_D2 = A2.slogdet()
        sl_D = sl_D1[0] * sl_D2[0], sl_D1[1] + sl_D2[1]
        sl_Psi = slog_fn_exp(lambda D: (c * D).sum(dim=-1), sl_D, dim=-1)
        # The cofactor matrices are calculated here, because their calculation
        # in the backward pass would be a bottleneck when evaluating the
        # backward pass repeatedly which happens in the Laplacian evaluation.
        # This solution is hugely suboptimal if only forward pass is needed (by
        # ~1.5 order of magnitude), but that never happens in normal use of our code
        ctx.save_for_backward(
            c, A1, A2, *_slogcof(A1), *_slogcof(A2), *sl_Psi, *sl_D, *sl_D1, *sl_D2
        )
        return sl_Psi

    @staticmethod
    def backward(ctx, _, Pb):
        cb, A1b, A2b = SLogLinearDetBackward.apply(Pb, *ctx.saved_tensors)
        return cb, A1b, A2b


def _backward_sloglin(Pb, sl_c, sl_Psi, sl_D):
    sl_Pb = slog(Pb)
    idx = ..., None
    sl_PbPsi = sl_Pb[0] * sl_Psi[0], sl_Pb[1] - sl_Psi[1]
    cb = sl_PbPsi[0][idx] * sl_D[0] * (sl_PbPsi[1][idx] + sl_D[1]).exp()
    sl_Db = sl_PbPsi[0][idx] * sl_c[0], sl_PbPsi[1][idx] + sl_c[1]
    return cb, sl_Db


def _backward_det(sl_Db, sl_C):
    idx = ..., None, None
    Ab = sl_Db[0][idx] * sl_C[0] * (sl_Db[1][idx] + sl_C[1]).exp()
    return Ab


class SLogLinearDetBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Pb, *args):
        c, A1, A2, *args = args
        *sl_C1, U1, s1, V1, sgn_UV1 = args[:6]
        *sl_C2, U2, s2, V2, sgn_UV2 = args[6:12]
        sl_Psi, sl_D, sl_D1, sl_D2 = zip(args[12::2], args[13::2])
        sl_c = slog(c)
        cb, sl_Db = _backward_sloglin(Pb, sl_c, sl_Psi, sl_D)
        # backward through D = D1 * D2
        sl_D1b = sl_Db[0] * sl_D2[0], sl_Db[1] + sl_D2[1]
        sl_D2b = sl_Db[0] * sl_D1[0], sl_Db[1] + sl_D1[1]
        A1b = _backward_det(sl_D1b, sl_C1)
        A2b = _backward_det(sl_D2b, sl_C2)
        ctx.save_for_backward(
            Pb,
            *(U1, V1, s1, sgn_UV1),
            *(U2, V2, s2, sgn_UV2),
            *(*sl_Db, *sl_Psi, *sl_D, *sl_c),
            *(*sl_C1, *sl_D1, *sl_D1b),
            *(*sl_C2, *sl_D2, *sl_D2b),
        )
        return cb, A1b, A2b

    @staticmethod
    def backward(ctx, cbt, A1bt, A2bt):
        Pbt, ct, A1t, A2t = SLogLinearDetDoubleBackward.apply(
            cbt, A1bt, A2bt, *ctx.saved_tensors
        )
        return (Pbt, ct, A1t, A2t, *(20 * [None]))


def _double_backward_det(Abt, U, V, sgn_UV, s, sl_C, sl_Db):
    if Abt.shape[-1] == 0:
        return (torch.zeros_like(sgn_UV), torch.zeros_like(sgn_UV)), torch.zeros_like(U)
    if Abt.shape[-1] == 1:
        return slog(Abt[..., 0, 0]), torch.zeros_like(U)
    sl_Abt = slog(Abt)
    sl_Dbt = slog_fn_exp(
        lambda CAbt: CAbt.sum(dim=(-1, -2)),
        (sl_C[0] * sl_Abt[0], sl_C[1] + sl_Abt[1]),
        dim=(-1, -2),
    )
    M = V.transpose(-1, -2) @ Abt.transpose(-1, -2) @ U
    sl_M = slog(M)
    log_r = log_rho(s)
    sl_Xi = torch.empty_like(M), torch.empty_like(M)
    i, j = idx_perm(M.shape[-1], 2, M.device)
    sl_Xi[0][..., i, j] = -sl_M[0][..., i, j]
    sl_Xi[1][..., i, j] = sl_M[1][..., i, j] + log_r
    bdiag(sl_Xi[0])[...], bdiag(sl_Xi[1])[...] = slog_fn_exp(
        lambda Mr: Mr.sum(dim=-1),
        (bdiag(sl_M[0])[..., j], bdiag(sl_M[1])[..., j] + log_r),
        dim=-1,
    )
    idx = ..., None, None
    sl_Y = slog_fn_exp(
        lambda Xi: sgn_UV[idx] * U @ Xi @ V.transpose(-1, -2),
        sl_Xi,
        dim=(-1, -2),
        idx=idx,
    )
    At = sl_Db[0][idx] * sl_Y[0] * (sl_Db[1][idx] + sl_Y[1]).exp()
    return sl_Dbt, At


def _double_backward_sloglin(cbt, sl_Dbt, sl_D, sl_Psi, sl_c, Pb):
    sl_cbt = slog(cbt)
    sl_Dcbt = slog_fn_exp(
        lambda Dcbt: Dcbt.sum(dim=-1),
        (sl_D[0] * sl_cbt[0], sl_D[1] + sl_cbt[1]),
        dim=-1,
    )
    sl_cDbt = slog_fn_exp(
        lambda cDbt: cDbt.sum(dim=-1),
        (sl_c[0] * sl_Dbt[0], sl_c[1] + sl_Dbt[1]),
        dim=-1,
    )
    Pbt = (
        sl_Psi[0] * sl_Dcbt[0] * (-sl_Psi[1] + sl_Dcbt[1]).exp()
        + sl_Psi[0] * sl_cDbt[0] * (-sl_Psi[1] + sl_cDbt[1]).exp()
    )
    sl_Psi2Dcbt = -sl_Dcbt[0], -2 * sl_Psi[1] + sl_Dcbt[1]
    sl_Psi2cDbt = -sl_cDbt[0], -2 * sl_Psi[1] + sl_cDbt[1]
    idx = ..., None
    ct = Pb[idx] * (
        sl_Psi2Dcbt[0][idx] * sl_D[0] * (sl_Psi2Dcbt[1][idx] + sl_D[1]).exp()
        + sl_Psi2cDbt[0][idx] * sl_D[0] * (sl_Psi2cDbt[1][idx] + sl_D[1]).exp()
        + sl_Psi[0][idx] * sl_Dbt[0] * (-sl_Psi[1][idx] + sl_Dbt[1]).exp()
    )
    sl_Dt = slog_fn_exp(
        lambda x: Pb[idx] * x.sum(dim=-1),
        (sl_Psi2Dcbt[0][idx] * sl_c[0], sl_Psi2Dcbt[1][idx] + sl_c[1]),
        (sl_Psi2cDbt[0][idx] * sl_c[0], sl_Psi2cDbt[1][idx] + sl_c[1]),
        (sl_Psi[0][idx] * sl_cbt[0], -sl_Psi[1][idx] + sl_cbt[1]),
    )
    return Pbt, ct, sl_Dt


class SLogLinearDetDoubleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cbt, A1bt, A2bt, *args):
        Pb, U1, V1, s1, sgn_UV1, U2, V2, s2, sgn_UV2, *args = args
        sl_Db, sl_Psi, sl_D, sl_c, sl_C1, sl_D1, sl_D1b, sl_C2, sl_D2, sl_D2b = zip(
            args[::2], args[1::2]
        )
        sl_D1bt, A1t = _double_backward_det(A1bt, U1, V1, sgn_UV1, s1, sl_C1, sl_D1b)
        sl_D2bt, A2t = _double_backward_det(A2bt, U2, V2, sgn_UV2, s2, sl_C2, sl_D2b)
        # double backward through D = D1 * D2
        sl_Dbt = slog_fn_exp(
            lambda x: x.sum(dim=-1),
            (sl_D2[0] * sl_D1bt[0], sl_D2[1] + sl_D1bt[1]),
            (sl_D1[0] * sl_D2bt[0], sl_D1[1] + sl_D2bt[1]),
        )
        sl_D1t = sl_Db[0] * sl_D2bt[0], sl_Db[1] + sl_D2bt[1]
        sl_D2t = sl_Db[0] * sl_D1bt[0], sl_Db[1] + sl_D1bt[1]
        Pbt, ct, sl_Dt = _double_backward_sloglin(cbt, sl_Dbt, sl_D, sl_Psi, sl_c, Pb)
        # reverse to collect inputs-tilde
        # reverse through D = D1 * D2
        sl_D1t = slog_fn_exp(
            lambda x: x.sum(dim=-1), sl_D1t, (sl_D2[0] * sl_Dt[0], sl_D2[1] + sl_Dt[1])
        )
        sl_D2t = slog_fn_exp(
            lambda x: x.sum(dim=-1), sl_D2t, (sl_D1[0] * sl_Dt[0], sl_D1[1] + sl_Dt[1])
        )
        # reverse through Di = det Ai
        idx = ..., None, None
        A1t = A1t + sl_D1t[0][idx] * sl_C1[0] * (sl_D1t[1][idx] + sl_C1[1]).exp()
        A2t = A2t + sl_D2t[0][idx] * sl_C2[0] * (sl_D2t[1][idx] + sl_C2[1]).exp()
        return Pbt, ct, A1t, A2t


sloglindet = getattr(SLogLinearDet, 'apply', None)
# works also when torch is mocked by sphinx
