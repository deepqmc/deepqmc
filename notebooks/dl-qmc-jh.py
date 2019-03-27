#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
get_ipython().run_line_magic('config', "InlineBackend.print_figure_kwargs = {'bbox_inches': 'tight', 'dpi': 300}")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


get_ipython().run_line_magic('aimport', 'dlqmc.NN, dlqmc.Sampler, dlqmc.Examples')
get_ipython().run_line_magic('aimport', 'dlqmc.utils, dlqmc.sampling, dlqmc.analysis, dlqmc.gto, dlqmc.physics, dlqmc.nn, dlqmc.fit')


# In[94]:


from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import torch
import pyscf
from pyscf import gto, scf
from tqdm.auto import tqdm, trange
from tensorboardX import SummaryWriter

from dlqmc.nn import WFNet
from dlqmc.fit import fit_wfnet, loss_local_energy, wfnet_fit_driver
from dlqmc.sampling import samples_from, langevin_monte_carlo
from dlqmc.physics import local_energy
from dlqmc.gto import GTOWF
from dlqmc.analysis import autocorr_coeff, blocking
from dlqmc.utils import (
    plot_func_xy, plot_func_x, integrate_on_mesh, form_geom, as_pyscf_atom
)


# In[5]:


h_atom = form_geom([[1, 0, 0]], [1])
h2_plus = form_geom([[-1, 0, 0], [1, 0, 0]], [1, 1])


# ## GTO WF

# In[6]:


mol = gto.M(
    atom=as_pyscf_atom(h2_plus),
    unit='bohr',
    basis='aug-cc-pv5z',
    charge=1,
    spin=1,
)
mf = scf.RHF(mol)
scf_energy_big = mf.kernel()
gtowf_big = GTOWF(mf, 0)


# In[7]:


mol = gto.M(
    atom=as_pyscf_atom(h2_plus),
    unit='bohr',
    basis='6-311g',
    charge=1,
    spin=1,
)
mf = scf.RHF(mol)
scf_energy = mf.kernel()
gtowf = GTOWF(mf, 0, use_pyscf=False)


# In[8]:


plot_func_x(lambda x: pyscf.dft.numint.eval_ao(mol, x), [-7, 7], is_torch=False);
plt.ylim(-1, 1)


# In[9]:


plot_func_x(lambda x: pyscf.dft.numint.eval_ao(gtowf_big._mol, x), [-7, 7], is_torch=False);
plt.ylim(-1, 1)


# In[11]:


integrate_on_mesh(lambda x: gtowf(x.cuda())**2, [(-6, 6), (-4, 4), (-4, 4)])


# In[12]:


plot_func_x(lambda x: local_energy(x, gtowf, h2_plus)[0], [-3, 3])
plt.ylim((-10, 0));


# In[28]:


n_walker = 1_000
sampler = langevin_monte_carlo(
    gtowf,
    torch.randn(n_walker, 3),
    tau=0.1,
)
samples, info = samples_from(sampler, trange(500))
E_loc = local_energy(samples.view(-1, 3), gtowf, h2_plus)[0].view(n_walker, -1)
info.acceptance.mean()


# In[30]:


plt.plot(*samples[0][:50, :2].numpy().T)
plt.gca().set_aspect(1)


# In[31]:


plt.plot(E_loc.mean(dim=0).numpy())
plt.ylim(-0.7, -0.5)


# In[32]:


plt.hist2d(
    *samples[:, 50:].reshape(-1, 3)[:, :2].numpy().T,
    bins=100,
    range=[[-3, 3], [-3, 3]],
)                                   
plt.gca().set_aspect(1)


# In[33]:


plt.hist(E_loc[:, 50:].flatten().clamp(-1.25, 0).numpy(), bins=100);


# In[17]:


E_loc[:, 50:].std()


# In[34]:


scf_energy, E_loc[:, 50:].mean().item(), (E_loc[:, 50:].mean(dim=1).std()/np.sqrt(E_loc.shape[0])).item()


# In[35]:


plt.plot(blocking(E_loc[:, 50:]).numpy())


# In[36]:


plt.plot(autocorr_coeff(range(50), E_loc[:, 50:]).numpy())
plt.axhline()


# ## Net WF

# In[38]:


plot_func_x(lambda x: WFNet(h_atom, n_dist_feats=32)._featurize(x)[1], [1, 11]);


# In[73]:


wfnet = WFNet(h2_plus, n_dist_feats=32, ion_pot=0.7, alpha=1.)
wfnet.cuda()
sampler = langevin_monte_carlo(
    wfnet,
    torch.randn(1_000, 3, device='cuda'),
    tau=0.1,
)


# In[74]:


exp_label = 'exp21'
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
wfnet.cpu()


# In[75]:


plot_func_x(
    lambda x: local_energy(x, wfnet, wfnet.geom)[0],
    [-15, 15],
)
plt.ylim((-1, 0));


# In[76]:


plot_func_xy(
    lambda x: wfnet._nn(wfnet._featurize(x)[1]).squeeze(),
    [[-10, 10], [-10, 10]]
);


# In[80]:


n_walker = 1_000
sampler = langevin_monte_carlo(
    wfnet.cuda(),
    torch.randn(n_walker, 3, device='cuda'),
    tau=0.1,
)
samples, info = samples_from(sampler, trange(500))
E_loc = local_energy(samples.view(-1, 3), wfnet, wfnet.geom)[0].view(n_walker, -1)
samples = samples.cpu()
E_loc = E_loc.cpu()
wfnet.cpu()
info.acceptance.mean()


# In[81]:


plt.plot(E_loc.mean(dim=0).numpy())
plt.ylim(-0.7, -0.5)


# In[82]:


plt.hist(E_loc[:, 50:].flatten().clamp(-1.25, 0).numpy(), bins=100);
plt.xlim(-1.25, 0)


# In[83]:


E_loc[:, 50:].std()


# In[84]:


scf_energy_big, E_loc[:, 50:].mean().item(), (E_loc[:, 50:].mean(dim=1).std()/np.sqrt(E_loc.shape[0])).item()


# In[87]:


bounds = [-2, 2]
plot_func_x(lambda x: torch.log(gtowf_big(x)), bounds, label='~exact WF');
plot_func_x(lambda x: torch.log(gtowf(x)), bounds, label='small-basis WF');
plot_func_x(lambda x: torch.log(wfnet(x))-0.67, bounds, label='DL WF');
plot_func_x(lambda x: torch.log(wfnet._asymptote(wfnet._featurize(x)[0]))-0.85, bounds, label='asymptotics');
plot_func_x(lambda x: wfnet._nn(wfnet._featurize(x)[1]).squeeze()-0.7, bounds, label='NN');
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.ylim(-1.5, None)

