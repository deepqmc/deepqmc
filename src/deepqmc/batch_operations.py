import torch
import torch.distributed as dist


__all__ = ()


def batch_len(t, parallel=False):
    len_t = len(t)
    if parallel:
        len_t = torch.as_tensor(len_t, dtype=t.dtype, device=t.device)
        dist.all_reduce(len_t, dist.ReduceOp.SUM)
    return len_t


def batch_sum(t, parallel=False):
    t_sum = t.sum()
    if parallel:
        dist.all_reduce(t_sum, dist.ReduceOp.SUM)
    return t_sum


def batch_mean(t, parallel=False):
    return batch_sum(t, parallel) / batch_len(t, parallel)


def batch_median(t, parallel=False):
    t = batch_gather_and_concat(t, parallel)  # batch median needs all elements
    return t.median()


def batch_max(t, parallel=False):
    t_max = t.max()
    if parallel:
        dist.all_reduce(t_max, dist.ReduceOp.MAX)
    return t_max


def batch_min(t, parallel=False):
    t_min = t.min()
    if parallel:
        dist.all_reduce(t_min, dist.ReduceOp.MIN)
    return t_min


def batch_exp_normalize_mean(t, parallel=False):
    t_max = batch_max(t, parallel)
    t_shifted = t - t_max
    return t_shifted.exp() / batch_mean(t_shifted.exp(), parallel)


def batch_weighted_mean_var(t, log_ws= None, parallel=False):
    log_ws = torch.zeros_like(t) if log_ws is None else log_ws
    ws = batch_exp_normalize_mean(log_ws, parallel)
    mean = batch_mean(ws * t, parallel)
    return mean, batch_mean((ws * (t - mean) ** 2), parallel)


def batch_gather_and_concat(t, parallel=False):
    if parallel:
        tensor_list = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=tensor_list, tensor=t)
        t = torch.concat(tensor_list)
    return t
