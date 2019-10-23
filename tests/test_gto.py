import numpy as np
import pytest
import torch
from pyscf import dft, gto, scf
from pytest import approx
from torch.testing import assert_allclose

from dlqmc.nn import PauliNet, pairwise_diffs
from dlqmc.pyscfext import eval_ao_normed


@pytest.fixture(scope='module')
def mol():
    return gto.M(atom='H 0 0 0', basis='cc-pvqz', cart=True, spin=1)


@pytest.fixture(scope='module')
def grids(mol):
    return dft.gen_grid.Grids(mol).build()


@pytest.fixture(scope='module')
def mf(mol):
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    return mf


@pytest.fixture
def gtowf(mf):
    return PauliNet.from_pyscf(mf).double()


def test_eval_ao_normed(mol, grids):
    ovlps = (eval_ao_normed(mol, grids.coords) ** 2 * grids.weights[:, None]).sum(0)
    assert np.max(np.abs(ovlps - 1)) < 1e-8


def test_torch_gto_aos(gtowf, grids):
    coords, weights = map(torch.tensor, (grids.coords, grids.weights))
    ovlps = (
        gtowf.mo.basis(pairwise_diffs(coords, gtowf.coords)) ** 2 * weights[:, None]
    ).sum(dim=0)
    assert_allclose(ovlps, 1)


def test_torch_gto_density(gtowf, grids):
    coords, weights = map(torch.tensor, (grids.coords, grids.weights))
    n_elec = (gtowf(coords[:, None, :]) ** 2 * weights).sum()
    assert n_elec.item() == approx(1)
