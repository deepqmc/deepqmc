import torch
import torch.distributed as dist

PARALLEL = False

__all__ = ()


def batch_len(t):
    len_t = len(t)
    if PARALLEL:
        len_t = torch.as_tensor(len_t, dtype=t.dtype, device=t.device)
        dist.all_reduce(len_t, dist.ReduceOp.SUM)
    return len_t


def batch_sum(t):
    t_sum = t.sum()
    if PARALLEL:
        dist.all_reduce(t_sum, dist.ReduceOp.SUM)
    return t_sum


def batch_mean(t):
    return batch_sum(t) / batch_len(t)


def batch_median(t):
    t = batch_gather_and_concat(t)  # batch median needs all elements
    return t.median()


def batch_max(t):
    t_max = t.max()
    if PARALLEL:
        dist.all_reduce(t_max, dist.ReduceOp.MAX)
    return t_max


def batch_min(t):
    t_min = t.min()
    if PARALLEL:
        dist.all_reduce(t_min, dist.ReduceOp.MIN)
    return t_min


def batch_exp_normalize_mean(t):
    t_max = batch_max(t)
    t_shifted = t - t_max
    return t_shifted.exp() / batch_mean(t_shifted.exp())


def batch_weighted_mean_var(t, log_ws=None):
    log_ws = torch.zeros_like(t) if log_ws is None else log_ws
    ws = batch_exp_normalize_mean(log_ws)
    mean = batch_mean(ws * t)
    return mean, batch_mean((ws * (t - mean) ** 2))


def batch_gather_and_concat(t):
    if PARALLEL:
        tensor_list = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=tensor_list, tensor=t)
        t = torch.concat(tensor_list)
    return t
