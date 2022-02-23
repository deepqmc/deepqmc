import torch

__all__ = ()


def unused_cuda_memory():
    import subprocess

    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e6
    out = subprocess.run(['nvidia-smi', '-q'], capture_output=True).stdout.decode()
    mem_used = sum(int(l.split()[4]) for l in out.split('\n') if 'Used GPU Memory' in l)
    mem_used *= 1024**2 / 1e6
    return mem_total - mem_used
