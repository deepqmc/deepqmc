import re
from pathlib import Path
from typing import Optional

import h5py
import jax
import numpy as np


def gather_electron_axis(pytree, electron_batch_axis=4):
    r"""Gather the electron samples from the devices to a single axis."""
    return jax.tree_util.tree_map(
        lambda x: np.moveaxis(x, 1, electron_batch_axis - 1).reshape(
            x.shape[0],
            *x.shape[2:electron_batch_axis],
            -1,
            *x.shape[electron_batch_axis + 1 :],
        ),
        pytree,
    )


def subscript_sorting_key(string_with_subscript: str):
    r"""Extracts the integer subscript from strings such as foo_2."""
    re_match = re.search(r'.+_(\d+)', string_with_subscript)
    assert re_match, f'Invalid string with substring {string_with_subscript}'
    return int(re_match.group(1))


def is_multi_node_subdir(subdir_name: str):
    r"""Checks if a subdir name is of the form training_0, or evaluation_1, etc."""
    assert subdir_name.startswith('training') or subdir_name.startswith(
        'evaluation'
    ), f'Invalid subdir name {subdir_name}'
    return re.search(r'.+_\d+', subdir_name) is not None


def sorted_subdirs(subdirs: list[str]) -> list[str]:
    r"""Sorts subdirs with potential integer subscripts."""
    are_multi_node_subdir = [is_multi_node_subdir(subdir) for subdir in subdirs]
    if any(are_multi_node_subdir):
        assert all(are_multi_node_subdir), 'Mix of single and multi node subdirs'
        assert sorted([subscript_sorting_key(subdir) for subdir in subdirs]) == list(
            range(len(subdirs))
        ), 'Invalid subscripts for multi node subdirs'
        return sorted(subdirs, key=subscript_sorting_key)
    else:
        assert len(subdirs) == 1, 'Multiple single node subdirs found'
        return subdirs


def chkpt_file_iteration(chkpt_file_name: str):
    r"""Extract the iteration count from the name of a checkpoint file."""
    re_match = re.search(r'chkpt-(\d+).pt', chkpt_file_name)
    assert re_match, f'Invalid checkpoint file name: {chkpt_file_name}'
    return int(re_match.group(1))


def last_checkpoint_iteration(path: Path) -> Optional[int]:
    r"""Return the iteration of the last checkpoint file in a deepQMC subdir."""
    chkpt_iterations = sorted(
        [chkpt_file_iteration(file.name) for file in path.glob('chkpt-*.pt')]
    )
    if len(chkpt_iterations) > 0:
        return chkpt_iterations[-1]
    return None


def concatenate_subdir_results(
    subdir_results: list[tuple[dict, Optional[int]]],
) -> tuple[dict, Optional[int]]:
    r"""Concatenate results from multiple deepQMC subdirs."""
    if len(subdir_results) == 1:
        return subdir_results[0]
    results, last_chkpt_iters = zip(*subdir_results)
    assert all(
        last_chkpt_iter == last_chkpt_iters[0]
        for last_chkpt_iter in last_chkpt_iters[1:]
    ), 'Mismatching last checkpoint iterations between subdirs'
    assert all(
        result.keys() == results[0].keys() for result in results[1:]
    ), 'Mismatching keys between subdirs'
    min_lengths = {
        key: min(len(result[key]) for result in results) for key in results[0].keys()
    }
    results = {
        key: (
            results[0][key]
            if 'samples' not in key
            else np.concatenate(
                [result[key][: min_lengths[key]] for result in results], axis=1
            )
        )
        for key in results[0].keys()
    }
    return results, last_chkpt_iters[0]


def read_subdir(path: Path, keys: list[str]) -> tuple[dict, Optional[int]]:
    r"""Read values of given keys from a result.h5 file in a deepQMC subdir."""
    last_chkpt_iter = last_checkpoint_iteration(path)
    result_file = path / 'result.h5'
    if not result_file.exists():
        return {}, None
    with h5py.File(result_file, swmr=True, libver='v110') as f:
        results = {key: np.array(f[key]) for key in keys if key in f.keys()}
    return results, last_chkpt_iter


def read_workdir(path: Path, keys: list[str]) -> tuple[dict, Optional[int]]:
    r"""Read values of given keys from result.h5 files in a deepQMC workdir."""
    eval_subdirs = [subdir.name for subdir in path.glob('evaluation*')]
    train_subdirs = [subdir.name for subdir in path.glob('training*')]
    if not eval_subdirs and not train_subdirs:
        return {}, None
    if eval_subdirs and train_subdirs:
        raise ValueError(
            f'workdir {path} contains both evaluation and training subdirs:'
            f' {eval_subdirs + train_subdirs}'
        )
    subdirs = eval_subdirs if not train_subdirs else train_subdirs
    subdir_results = [
        read_subdir(path / subdir, keys) for subdir in sorted_subdirs(subdirs)
    ]
    workdir_result, last_chkpt_iter = concatenate_subdir_results(subdir_results)
    return workdir_result, last_chkpt_iter


def convert_to_per_molecule_format(
    raw_result: np.ndarray, mol_idxs: np.ndarray
) -> np.ndarray:
    r"""Convert results (local energies, psi values, etc.) to per molecule format.

    Args:
        raw_result [n_iter, molecule_batch_size, ...]: the result
            values in batched format used during training/evaluation.
        mol_idxs [n_iter, molecule_batch_size]: the global dataset indices of the
            molecules considered in each iteration.

    Returns:
        [n_iter_per_molecule, n_molecules, ...]: the results
            rearranged into per molecule format.
    """
    mol_idxs = mol_idxs.astype(int)
    quantity_shape = raw_result.shape[2:]
    n_mol = mol_idxs.max() + 1
    steps_per_mol = mol_idxs.size // n_mol
    even_steps = steps_per_mol * n_mol

    mol_idx = mol_idxs.flatten()[:even_steps]
    result = raw_result.reshape(-1, *quantity_shape)[:even_steps]
    cumulative_idx_per_mol = (
        np.cumsum(mol_idx[..., None] == np.arange(n_mol), axis=0) - 1
    )
    step_idx_per_mol = cumulative_idx_per_mol[np.arange(len(mol_idx)), mol_idx]
    result_per_mol = np.zeros((steps_per_mol, n_mol, *quantity_shape))
    result_per_mol[step_idx_per_mol, mol_idx] = result
    return result_per_mol


def read_and_convert_result(
    path, *keys, read_workdir=read_workdir, gather_electrons=True
):
    r"""Read and convert results from a deepQMC workdir to per molecule format."""
    results, _ = read_workdir(path, [*keys, 'mol_idxs'])
    min_idxs = {
        key: min(len(result) for result in results.values()) for key in results.keys()
    }
    electrons_gathered = {
        key: (
            gather_electron_axis(results[key][: min_idxs[key]])
            if gather_electrons
            else results[key][: min_idxs[key], 0]
        )
        for key in keys
    }
    results = {
        k: convert_to_per_molecule_format(
            electrons_gathered[k],
            results['mol_idxs'][: min_idxs[k]],
        )
        for k in keys
    }
    return list(results.values())[0] if len(results.keys()) == 1 else results
