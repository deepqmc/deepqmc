import torch

from ..errors import DeepQMCError

__all__ = ()


def unused_cuda_memory():
    import subprocess

    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e6
    out = subprocess.run(['nvidia-smi', '-q'], capture_output=True).stdout.decode()
    mem_used = sum(int(l.split()[4]) for l in out.split('\n') if 'Used GPU Memory' in l)
    mem_used *= 1024 ** 2 / 1e6
    return mem_total - mem_used


def estimate_optimal_batch_size_cuda(
    test_func, test_batch_sizes, mem_margin=0.9, max_memory=None
):
    assert len(test_batch_sizes) >= 4
    test_batch_sizes = torch.as_tensor(test_batch_sizes).float()
    mem = []
    for size in test_batch_sizes.int():
        torch.cuda.reset_max_memory_allocated()
        test_func(size.item())
        mem.append(torch.cuda.max_memory_allocated() / 1e6)
    mem = torch.tensor(mem)
    delta = (mem[1:] - mem[:-1]) / (test_batch_sizes[1:] - test_batch_sizes[:-1])
    delta = delta[1:]  # first try may be off due to caching
    assert (delta > 0).all()
    memory_per_batch = delta.mean() / mem_margin
    if delta.std() / memory_per_batch > 0.3:
        raise DeepQMCError(
            'Inconsistent estimation of GPU memory per batch. '
            'Try specifying large test_batch_sizes.'
        )
    max_memory = max_memory or unused_cuda_memory()
    return int(max_memory / memory_per_batch)
