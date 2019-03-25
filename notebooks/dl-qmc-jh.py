#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1')
get_ipython().run_line_magic('aimport', 'dlqmc.NN, dlqmc.Sampler, dlqmc.Examples')
get_ipython().run_line_magic('aimport', 'dlqmc.utils, dlqmc.sampling, dlqmc.analysis, dlqmc.gto, dlqmc.physics, dlqmc.nn')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
get_ipython().run_line_magic('config', "InlineBackend.print_figure_kwargs = {'bbox_inches': 'tight', 'dpi': 300}")


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from collections import namedtuple

import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.distributions import Normal
import pyscf
from pyscf import gto, scf, dft
from pyscf.data.nist import BOHR
from tqdm import tqdm_notebook, tnrange
from tensorboardX import SummaryWriter

from dlqmc.Sampler import HMC
from dlqmc.nn import WFNet
import dlqmc.nn
from dlqmc.sampling import langevin_monte_carlo
from dlqmc.utils import (
    plot_func, get_flat_mesh, assign_where, plot_func_xy,
    plot_func_x, integrate_on_mesh, form_geom, as_pyscf_atom
)
from dlqmc.physics import (
    local_energy, grad, quantum_force, nuclear_cusps, nuclear_energy
)
from dlqmc.gto import GTOWF
from dlqmc.analysis import autocorr_coeff, blocking


# In[4]:


h_atom = form_geom([[1, 0, 0]], [1])
h2_plus = form_geom([[-1, 0, 0], [1, 0, 0]], [1, 1])


# ## GTO WF

# In[5]:


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


# In[6]:


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


# In[7]:


plot_func_x(lambda x: pyscf.dft.numint.eval_ao(mol, x), [-7, 7], is_torch=False);
plt.ylim(-1, 1)


# In[8]:


plot_func_x(lambda x: pyscf.dft.numint.eval_ao(gtowf_big._mol, x), [-7, 7], is_torch=False);
plt.ylim(-1, 1)


# In[10]:


integrate_on_mesh(lambda x: gtowf(x.cuda())**2, [(-6, 6), (-4, 4), (-4, 4)])


# In[11]:


plot_func_x(lambda x: local_energy(x, gtowf, h2_plus)[0], [-3, 3])
plt.ylim((-10, 0));


# In[12]:


tau = 0.1
n_walker = 1_000
n_steps = 500
samples, info = langevin_monte_carlo(
    gtowf,
    torch.randn(n_walker, 3),
    n_steps,
    tau,
    range=tnrange
)
E_loc = local_energy(samples.view(-1, 3), gtowf, h2_plus)[0].view(n_walker, -1)
info


# In[13]:


plt.plot(*samples[0][:50, :2].numpy().T)
plt.gca().set_aspect(1)


# In[14]:


plt.plot(E_loc.mean(dim=0).numpy())
plt.ylim(-0.7, -0.5)


# In[15]:


plt.hist2d(
    *samples[:, 50:].reshape(-1, 3)[:, :2].numpy().T,
    bins=100,
    range=[[-3, 3], [-3, 3]],
)                                   
plt.gca().set_aspect(1)


# In[16]:


plt.hist(E_loc[:, 50:].flatten().clamp(-1.25, 0).numpy(), bins=100);


# In[17]:


E_loc[:, 50:].std()


# In[18]:


scf_energy, E_loc[:, 50:].mean().item(), (E_loc[:, 50:].mean(dim=1).std()/np.sqrt(E_loc.shape[0])).item()


# In[19]:


plt.plot(blocking(E_loc[:, 50:]).numpy())


# In[20]:


plt.plot(autocorr_coeff(range(50), E_loc[:, 50:]).numpy())
plt.axhline()


# ## Net WF

# In[21]:


plot_func_x(lambda x: WFNet(h_atom, n_dist_basis=32)._featurize(x)[1], [1, 11]);


# In[6]:


# rs_dl = iter(DataLoader(samples.view(-1, 3), batch_size=10_000, shuffle=True))


# In[22]:


def fit(wfnet, E_ref=None, n_steps=100, range=range, lr=0.005, writer=None):
    device = wfnet.ion_pot.device
    opt = torch.optim.Adam(wfnet.parameters(), lr=lr)
    for i_step in range(n_steps):
        rs = 3*torch.randn(10_000, 3, device=device)
        Es_loc, psis = local_energy(rs, wfnet, wfnet.geom, create_graph=True)
        ws = psis**2/psis.detach()**2
        # ws = torch.tensor(1.)
        E0 = E_ref if E_ref is not None else Es_loc.mean()
        loss = (ws*(Es_loc-E0)**2).mean()/ws.mean()
        if writer:
            writer.add_scalar('loss', loss, i_step)
            writer.add_scalar('E_loc/mean', Es_loc.mean(), i_step)
            writer.add_scalar('E_loc/var', Es_loc.var(), i_step)
            writer.add_scalar('param/ion_pot', wfnet.ion_pot, i_step)
            plot_func_x(
                lambda x: wfnet._nn(wfnet._featurize(x)[1]).squeeze(),
                [-15, 15],
                device=device,
            );
            writer.add_figure('x_line', plt.gcf(), i_step)
            plt.close()
        loss.backward()
        opt.step()
        opt.zero_grad()


# In[23]:


wfnet = WFNet(h2_plus, n_dist_basis=32, ion_pot=0.7, alpha=1.)


# In[24]:


writer = SummaryWriter('runs/exp6/pretrain')
wfnet.cuda()
fit(wfnet, E_ref=-0.5, n_steps=2000, writer=writer, range=tnrange)
wfnet.cpu()
writer.close()


# In[39]:


writer = SummaryWriter('runs/exp5/variance3')
wfnet.cuda()
fit(wfnet, E_ref=None, n_steps=300, writer=writer, range=tnrange)
wfnet.cpu()
writer.close()


# In[40]:


plot_func_x(
    lambda x: local_energy(x, wfnet, wfnet.geom)[0],
    [-15, 15],
)
plt.ylim((-1, 0));


# In[41]:


plot_func_xy(
    lambda x: wfnet._nn(wfnet._featurize(x)[1]).squeeze(),
    [[-10, 10], [-10, 10]]
);


# In[42]:


tau = 0.1
n_walker = 1_000
n_steps = 500
wfnet.cuda()
samples, info = langevin_monte_carlo(
    wfnet,
    torch.randn(n_walker, 3).cuda(),
    n_steps,
    tau,
    range=tnrange
)
E_loc = local_energy(samples.view(-1, 3), wfnet, wfnet.geom)[0].view(n_walker, -1)
samples = samples.cpu()
E_loc = E_loc.cpu()
wfnet.cpu()
info


# In[43]:


plt.plot(E_loc.mean(dim=0).numpy())
plt.ylim(-0.7, -0.5)


# In[55]:


plt.hist(E_loc[:, 50:].flatten().clamp(-1.25, 0).numpy(), bins=100);
plt.xlim(-1.25, 0)


# In[44]:


E_loc[:, 50:].std()


# In[45]:


scf_energy_big, E_loc[:, 50:].mean().item(), (E_loc[:, 50:].mean(dim=1).std()/np.sqrt(E_loc.shape[0])).item()


# In[58]:


bounds = [-3, 3]
plot_func_x(lambda x: gtowf(x), bounds);
plot_func_x(lambda x: gtowf_big(x), bounds);
plt.ylim(0, None)


# In[63]:


bounds = [-2, 2]
plot_func_x(lambda x: torch.log(gtowf_big(x)), bounds, label='~exact WF');
plot_func_x(lambda x: torch.log(gtowf(x)), bounds, label='small-basis WF');
plot_func_x(lambda x: torch.log(wfnet(x))-0.76, bounds, label='DL WF');
plot_func_x(lambda x: torch.log(wfnet._asymptote(wfnet._featurize(x)[0]))-0.85, bounds, label='asymptotics');
plot_func_x(lambda x: wfnet._nn(wfnet._featurize(x)[1]).squeeze()-0.7, bounds, label='NN');
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2)
plt.ylim(-1.5, None)

