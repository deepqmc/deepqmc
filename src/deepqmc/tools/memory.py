import collections

import torch

__all__ = ()

EMPTY_TYPES = (str, type(None), type({}.keys()))
UNKNWON_CLASSES = set()


def get_children(obj):
    if type(obj) in (dict, collections.defaultdict):
        return obj.items()
    if type(obj) is list or isinstance(obj, tuple):
        return ((i, v) for i, v in enumerate(obj))
    if type(obj) in (set, frozenset, collections.deque):
        return (('?', v) for v in obj)
    try:
        return obj.__dict__.items()
    except AttributeError:
        pass
    UNKNWON_CLASSES.add(str(type(obj)))
    return ()


def find_large_cuda_tensors(obj, depth=False, threshold=1e6):
    visited = set()
    queue = collections.deque()
    queue.append((obj, ''))
    while queue:
        n, addr = queue.pop() if depth else queue.popleft()
        visited.add(id(n))
        if torch.is_tensor(n) and n.is_cuda and n.numel() > threshold:
            print(addr, type(n), n.shape)
        queue.extend(
            (v, f'{addr}.{k}') for k, v in get_children(n) if id(v) not in visited
        )
