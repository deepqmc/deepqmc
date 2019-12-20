from functools import lru_cache
from itertools import combinations, permutations

import numpy as np

__all__ = ()


@lru_cache()
def pair_idxs(n, unique=False):
    return np.array(list((combinations if unique else permutations)(range(n), 2)))


@lru_cache()
def spin_pair_idxs(n_up, n_down, unique=False, transposed=False, joined=False):
    # indexes for up-up, up-down, down-up, down-down
    ij = pair_idxs(n_up + n_down, unique=unique)
    mask = ij < n_up
    idxs = (
        ij[mask.all(axis=1)],
        ij[np.diff(mask.astype(int))[:, 0] == -1],
        ij[np.diff(mask.astype(int))[:, 0] == 1],
        ij[~mask.any(axis=1)],
    )
    if joined:
        idxs = np.concatenate([idxs[0], idxs[3]]), np.concatenate([idxs[1], idxs[2]])
    if transposed:
        idxs = tuple(idx.T for idx in idxs)
    return idxs
