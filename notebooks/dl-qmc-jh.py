#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
get_ipython().run_line_magic('config', "InlineBackend.print_figure_kwargs = {'bbox_inches': 'tight', 'dpi': 300}")
get_ipython().run_line_magic('matplotlib', 'inline')


get_ipython().run_line_magic('aimport', 'dlqmc.utils, dlqmc.sampling, dlqmc.analysis, dlqmc.gto,     dlqmc.physics, dlqmc.nn, dlqmc.fit, dlqmc.torchext, dlqmc.geom')


import time
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import torch
from torch import nn
import pyscf
from pyscf import gto, scf
from pyscf.data.nist import BOHR
from tqdm.auto import tqdm, trange
from tensorboardX import SummaryWriter

from dlqmc.nn import WFNet, DistanceBasis
from dlqmc.fit import fit_wfnet, loss_local_energy, wfnet_fit_driver
from dlqmc.sampling import samples_from, langevin_monte_carlo
from dlqmc.physics import local_energy
from dlqmc.gto import TorchGTOSlaterWF, PyscfGTOSlaterWF
from dlqmc.analysis import autocorr_coeff, blocking
from dlqmc.geom import Geometry
from dlqmc.utils import plot_func, plot_func_xy, plot_func_x, integrate_on_mesh
from dlqmc import torchext


h_atom = Geometry([[1, 0, 0]], [1])
h2_plus = Geometry([[-1, 0, 0], [1, 0, 0]], [1, 1])
h2_mol = Geometry([[0, 0, 0], [0.742/BOHR, 0, 0]], [1, 1])
be_atom = Geometry([[0, 0, 0]], [4])


# ## H2+

# ### GTO WF

mol = gto.M(
    atom=h2_plus.as_pyscf(),
    unit='bohr',
    basis='aug-cc-pv5z',
    charge=1,
    spin=1,
)
mf = scf.RHF(mol)
scf_energy_big = mf.kernel()
gtowf_big = PyscfGTOSlaterWF(mf)


mol = gto.M(
    atom=h2_plus.as_pyscf(),
    unit='bohr',
    basis='6-311g',
    charge=1,
    spin=1,
)
mf = scf.RHF(mol)
scf_energy = mf.kernel()
gtowf = TorchGTOSlaterWF(mf)


plot_func_x(lambda x: pyscf.dft.numint.eval_ao(mol, x), [-7, 7], is_torch=False);
plt.ylim(-1, 1)


plot_func_x(lambda x: pyscf.dft.numint.eval_ao(gtowf_big._mol, x), [-7, 7], is_torch=False);
plt.ylim(-1, 1)


integrate_on_mesh(lambda x: gtowf(x.cuda()[:, None])**2, [(-6, 6), (-4, 4), (-4, 4)])


plot_func_x(lambda x: local_energy(x[:, None], gtowf, h2_plus)[0], [-3, 3])
plt.ylim((-10, 0));


n_walker = 1_000
sampler = langevin_monte_carlo(
    gtowf,
    torch.randn(n_walker, 1, 3),
    tau=0.1,
)
samples, info = samples_from(sampler, trange(500))
E_loc = local_energy(samples.flatten(end_dim=1), gtowf, h2_plus)[0].view(n_walker, -1)
info.acceptance.mean()


plt.plot(*samples[0][:50, 0, :2].numpy().T)
plt.gca().set_aspect(1)


plt.plot(E_loc.mean(dim=0).numpy())
plt.ylim(-0.7, -0.5)


plt.hist2d(
    *samples[:, 50:].flatten(end_dim=1)[:, 0, :2].numpy().T,
    bins=100,
    range=[[-3, 3], [-3, 3]],
)                                   
plt.gca().set_aspect(1)


plt.hist(E_loc[:, 50:].flatten().clamp(-1.25, 0).numpy(), bins=100);


E_loc[:, 50:].std()


scf_energy, E_loc[:, 50:].mean().item(), (E_loc[:, 50:].mean(dim=1).std()/np.sqrt(E_loc.shape[0])).item()


plt.plot(blocking(E_loc[:, 50:]).numpy())


plt.plot(autocorr_coeff(range(50), E_loc[:, 50:]).numpy())
plt.axhline()


# ### Net WF

plot_func(DistanceBasis(32), [1, 11]);


wfnet = WFNet(h2_plus, 1, ion_pot=0.7).cuda()
sampler = langevin_monte_carlo(
    wfnet,
    torch.randn(1_000, 1, 3, device='cuda'),
    tau=0.1,
)


exp_label = 'exp30'
with SummaryWriter(f'runs/{exp_label}/pretrain') as writer:
    fit_wfnet(
        wfnet,
        partial(loss_local_energy, E_ref=-0.5),
        torch.optim.Adam(wfnet.parameters(), lr=0.005),
        wfnet_fit_driver(
            sampler,
            samplings=range(1),
            n_epochs=5,
            n_sampling_steps=550,
            batch_size=10_000,
            n_discard=50,
            range_sampling=partial(trange, desc='sampling steps', leave=False),
            range_training=partial(trange, desc='training steps', leave=False),
        ),
        writer=writer,
    )
with SummaryWriter(f'runs/{exp_label}/variance') as writer:
    fit_wfnet(
        wfnet,
        loss_local_energy,
        torch.optim.Adam(wfnet.parameters(), lr=0.005),
        wfnet_fit_driver(
            sampler,
            samplings=trange(10, desc='samplings'),
            n_epochs=5,
            n_sampling_steps=550,
            batch_size=10_000,
            n_discard=50,
            range_sampling=partial(trange, desc='sampling steps', leave=False),
            range_training=partial(trange, desc='training steps', leave=False),
        ),
        writer=writer,
    )
wfnet.cpu();


plot_func_x(
    lambda x: local_energy(x[:, None], wfnet, wfnet.geom)[0],
    [-15, 15],
)
plt.ylim((-1, 0));


plot_func_xy(
    lambda x: wfnet.deep_lin(wfnet._featurize(x[:, None])[0]).squeeze(),
    [[-10, 10], [-10, 10]]
);


n_walker = 1_000
sampler = langevin_monte_carlo(
    wfnet.cuda(),
    torch.randn(n_walker, 1, 3, device='cuda'),
    tau=0.1,
)
samples, info = samples_from(sampler, trange(500))
E_loc = local_energy(samples.flatten(end_dim=1), wfnet, wfnet.geom)[0].view(n_walker, -1)
samples = samples.cpu()
E_loc = E_loc.cpu()
wfnet.cpu()
info.acceptance.mean()


plt.plot(E_loc.mean(dim=0).numpy())
plt.ylim(-0.7, -0.5)


plt.hist(E_loc[:, 50:].flatten().clamp(-1.25, 0).numpy(), bins=100);
plt.xlim(-1.25, 0)


E_loc[:, 50:].std()


scf_energy_big, E_loc[:, 50:].mean().item(), (E_loc[:, 50:].mean(dim=1).std()/np.sqrt(E_loc.shape[0])).item()


bounds = [-2, 2]
plot_func_x(lambda x: torch.log(gtowf_big(x[:, None])), bounds, label='~exact WF');
plot_func_x(lambda x: torch.log(gtowf(x[:, None])), bounds, label='small-basis WF');
plot_func_x(lambda x: torch.log(wfnet(x[:, None]))-0.2, bounds, label='DL WF');
plot_func_x(lambda x: torch.log(wfnet.nuc_asymp(wfnet._featurize(x[:, None])[1][0]))-0.89, bounds, label='asymptotics');
plot_func_x(lambda x: wfnet.deep_lin(wfnet._featurize(x[:, None])[0]).squeeze()-0.2, bounds, label='NN');
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.ylim(-1.5, None)


# ## H2

# ### GTO WF

mol = gto.M(
    atom=h2_mol.as_pyscf(),
    unit='bohr',
    basis='6-311g',
    charge=0,
    spin=0,
)
mf = scf.RHF(mol)
mf.kernel()
gtowf = PyscfGTOSlaterWF(mf)


mol = gto.M(
    atom=h2_mol.as_pyscf(),
    unit='bohr',
    basis='6-311g',
    charge=0,
    spin=2,
)
mf = scf.RHF(mol)
mf.kernel()
gtowf = TorchGTOSlaterWF(mf)


rs = torch.randn(100, 2, 3)


gtowf(rs)


# ### Net WF

mol = gto.M(
    atom=as_pyscf_atom(h2_mol),
    unit='bohr',
    basis='6-311g',
    charge=1,
    spin=1,
)
mf = scf.RHF(mol)
scf_energy = mf.kernel()
gtowf = GTOWF(mf, 0, use_pyscf=False)


wfnet = WFNet(h2_mol, 2, ion_pot=0.7).cuda()
sampler = langevin_monte_carlo(
    wfnet,
    torch.randn(1_000, 2, 3, device='cuda'),
    tau=0.1,
)


exp_label = 'H2/exp01'
with SummaryWriter(f'runs/{exp_label}/pretrain') as writer:
    fit_wfnet(
        wfnet,
        partial(loss_local_energy, E_ref=-1.1),
        torch.optim.Adam(wfnet.parameters(), lr=0.005),
        wfnet_fit_driver(
            sampler,
            samplings=range(1),
            n_epochs=5,
            n_sampling_steps=550,
            batch_size=10_000,
            n_discard=50,
            range_sampling=partial(trange, desc='sampling steps', leave=False),
            range_training=partial(trange, desc='training steps', leave=False),
        ),
        writer=writer,
    )
with SummaryWriter(f'runs/{exp_label}/variance') as writer:
    fit_wfnet(
        wfnet,
        loss_local_energy,
        torch.optim.Adam(wfnet.parameters(), lr=0.005),
        wfnet_fit_driver(
            sampler,
            samplings=trange(10, desc='samplings'),
            n_epochs=5,
            n_sampling_steps=550,
            batch_size=10_000,
            n_discard=50,
            range_sampling=partial(trange, desc='sampling steps', leave=False),
            range_training=partial(trange, desc='training steps', leave=False),
        ),
        writer=writer,
    )
wfnet.cpu();


n_walker = 1_000
sampler = langevin_monte_carlo(
    wfnet.cuda(),
    torch.randn(n_walker, 2, 3, device='cuda'),
    tau=0.1,
)
samples, info = samples_from(sampler, trange(500))
E_loc = local_energy(samples.flatten(end_dim=1), wfnet, wfnet.geom)[0].view(n_walker, -1)
samples = samples.cpu()
E_loc = E_loc.cpu()
wfnet.cpu()
info.acceptance.mean()


plt.plot(E_loc.mean(dim=0).numpy())


def f(loops, steps):
    for loop in loops:
        for step in steps:
            time.sleep(0.1)


f(range(2), range(10))
f(trange(2), trange(10, leave=False))


# ## Be atom

mol = gto.M(
    atom=be_atom.as_pyscf(),
    unit='bohr',
    basis='6-311g',
    charge=0,
    spin=0,
)
mf = scf.RHF(mol)
mf.kernel()
gtowf = PyscfGTOSlaterWF(mf)


rs = torch.rand(100, 4, 3)


gtowf(rs)

