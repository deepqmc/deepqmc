from ..fit import WaveFunctionLoss

__all__ = ()


class LossVariance(WaveFunctionLoss):
    def forward(self, Es_loc, psis, ws, E_ref=None, p=1):
        assert psis.grad_fn is None
        E0 = E_ref if E_ref is not None else (ws * Es_loc).mean()
        return (ws * (Es_loc - E0).abs() ** p).mean()


def loss_least_squares(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()


def outlier_mask(x, p, q, dim=None):
    x = x.detach()
    dim = dim if dim is not None else -1
    n = x.shape[dim]
    lb = x.kthvalue(int(p * n), dim=dim).values
    ub = x.kthvalue(int((1 - p) * n), dim=dim).values
    return (
        (x - (lb + ub).unsqueeze(dim) / 2).abs() > q * (ub - lb).unsqueeze(dim),
        (lb, ub),
    )


def fit_wf_supervised(
    fit_net,
    true_net,
    loss_func,
    opt,
    sample_gen,
    correlated_sampling=True,
    acc_grad=1,
    writer=None,
    start=0,
    scheduler=None,
    epoch_size=100,
):
    for step, (rs, _) in enumerate(sample_gen, start=start):
        psis_fit = fit_net(rs)
        psis_true = true_net(rs)
        loss = loss_func(psis_fit, psis_true)
        if writer:
            writer.add_scalar('loss', loss, step)
            for label, value in fit_net.tracked_parameters():
                writer.add_scalar(f'param/{label}', value, step)
        loss.backward()
        if (step + 1) % acc_grad == 0:
            opt.step()
            fit_net.zero_grad()
        if scheduler and (step + 1) % epoch_size == 0:
            scheduler.step()
