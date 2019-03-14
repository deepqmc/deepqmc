#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1')
get_ipython().run_line_magic('aimport', 'dlqmc.NN, dlqmc.Sampler, dlqmc.Examples')
get_ipython().run_line_magic('aimport', 'dlqmc.utils, dlqmc.sampling, dlqmc.analysis, dlqmc.gto, dlqmc.physics, dlqmc.nn')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
get_ipython().run_line_magic('config', "InlineBackend.print_figure_kwargs = {'bbox_inches': 'tight', 'dpi': 300}")


# In[354]:


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

from dlqmc.Sampler import HMC
from dlqmc.nn import WFNet
import dlqmc.nn
from dlqmc.sampling import langevin_monte_carlo
from dlqmc.utils import (
    plot_func, get_flat_mesh, assign_where, plot_func_xy,
    plot_func_x, integrate_on_mesh
)
from dlqmc.physics import (
    local_energy, grad, quantum_force, nuclear_cusps, Geometry,
    nuclear_energy
)
from dlqmc.gto import GTOWF
from dlqmc.analysis import autocorr_coeff, blocking


# In[207]:


h_atom = Geometry([[1., 0., 0.]], [1.])
he_plus = Geometry([[1., 0., 0.]], [2.])
h2_plus = Geometry([[-1., 0., 0.], [1., 0., 0.]], [1., 1.])
he2_3plus = Geometry([[-1., 0., 0.], [1., 0., 0.]], [2., 2.])


# ## Net WF

# In[373]:


def cutoff(x, cutoff):
    x_rel = x/cutoff
    return 1-6*x_rel**5+15*x_rel**4-10*x_rel**3
    
for q in np.linspace(0, 1, 10):
    plot_func(
        lambda x: cutoff(x, 10)*np.exp(-(7/(1+10*q))**2*(x-10*q**2)**2),
        [0, 10]
    )


# In[375]:


tau = 0.1
n_walker = 1_000
n_steps = 500
samples, info = langevin_monte_carlo(
    net,
    torch.randn(n_walker, 3),
    n_steps,
    tau,
    range=tnrange
)
info


# In[376]:


rs_dl = iter(DataLoader(samples.view(-1, 3), batch_size=10_000, shuffle=True))


# In[380]:


plot_func_xy(
    lambda x: wf._nn(wf._featurize(x)[1]).squeeze(),
    [[-3, 3], [-2, 2]]
);


# In[669]:


plot_func_x(
    lambda x: torch.exp(wfnet._nn(net._featurize(x)[1])).squeeze(),
    [-6, 6]
)


# In[383]:


plot_func_x(wf, [-3, 3]);
plt.ylim(0, 1.5);


# In[660]:


def fit(wf, geom, E_ref, n_steps=100, range=range, lr=0.005):
    opt = torch.optim.Adam(wf.parameters(), lr=lr)
    for i in range(n_steps):
        rs = 3*torch.randn(10_000, 3)
        Es_loc, psis = local_energy(rs, wf, geom, create_graph=True)
        ws = psis**2/psis.detach()**2
        loss = (ws*(Es_loc-E_ref)**2).mean()/ws.mean()
        print(*(x.item() for x in (
            loss, Es_loc.mean(), Es_loc.var(), wf._ion_pot
        )))
        loss.backward()
        opt.step()
        opt.zero_grad()


# In[378]:


wf = WFNet(h2_plus, ion_pot=0.5)


# In[667]:


fit(wf, h2_plus, -0.6, n_steps=100)


# In[636]:


rs = 3*torch.randn(10_000, 3)
# rs = next(rs_dl)
E_loc, psi = local_energy(rs, net, h2_plus, create_graph=True)
# E_loc.mean(), E_loc.var()


# In[638]:


psi


# In[633]:


loss = ((E_loc+0.602)**2).sum()
loss


# In[634]:


loss.backward()
opt.step()
opt.zero_grad()


# In[635]:


E_loc = local_energy(rs, net, h2_plus)
E_loc.mean(), E_loc.var(), ((E_loc+0.602)**2).sum()


# In[477]:


plt.hist(E_loc.detach().clamp(-1.25, 1).numpy(), bins=100);


# In[29]:


plt.plot(
    x_line[:, 0].numpy(),
    local_energy(lambda x: net(x, coords), x_line, coords, charges).detach().numpy()
)
plt.ylim((-10, 0));


# In[30]:


samples = HMC_ad(
    dist=lambda x: net(x, coords).squeeze()**2,
    stepsize=0.1,
    dysteps=3,
    n_walker=100,
    steps=5000,
    dim=3,
    push=1,
    startfactor=1,
    presteps=50,
).detach().reshape(-1, 3)


# In[31]:


plt.hist2d(
    samples[:, 0].numpy(),
    samples[:, 1].numpy(),
    bins=100,
    range=[[-3, 3], [-3, 3]],
)                                   
plt.gca().set_aspect(1)


# In[32]:


E_loc = local_energy(lambda x: net(x, coords), samples, coords, charges)


# In[33]:


plt.hist(E_loc.detach().clamp(-1.25, 1).numpy(), bins=100);


# In[35]:


E_loc.mean().item()


# ## GTO WF

# In[333]:


mol = gto.M(
    atom=[
        ['H', (-1, 0, 0)],
        ['H', (1, 0, 0)]
    ],
    unit='bohr',
    basis='aug-cc-pv5z',
    charge=1,
    spin=1,
)
mf2 = scf.RHF(mol)
scf_energy = mf2.kernel()
gtowf2 = GTOWF(mf2, 0)


# In[ ]:


gtowf2


# In[347]:


plt.plot(gtowf._mo_coeffs[:10])


# In[372]:


(gtowf2._mo_coeffs > 1e-3).nonzero()


# In[370]:


mol = gto.M(
    atom=[
        [1, (-1, 0, 0)],
    ],
    unit='bohr',
    basis='aug-cc-pv5z',
    charge=0,
    spin=1,
)
mf = scf.RHF(mol)
scf_energy = mf.kernel()
gtowf = GTOWF(mf, 0)


# In[364]:


(gtowf._mo_coeffs > 1e-4).nonzero()


# In[ ]:



plot_func_x(lambda x: pyscf.dft.numint.eval_ao(gtowf._mol, x.numpy()), [-20, 20]);


# In[376]:


# plot_func_x(lambda x: torch.log(gtowf2(x)), [-20, 20]);
plot_func_x(lambda x: torch.log(gtowf(x)), [-20, 20]);
for i in range(6):
    plot_func_x(lambda x: np.log(gtowf._mo_coeffs[i] * pyscf.dft.numint.eval_ao(gtowf._mol, x)[:, i]), [-20, 20], is_torch=False);
plt.ylim(-20, 0)


# In[377]:


wfnet = WFNet(h2_plus, ion_pot=0.5, alpha=100.)
plot_func_x(lambda x: torch.log(gtowf2(x)), [-15, 15]);
plot_func_x(
    lambda x: torch.log(wfnet._asymptote(wfnet._featurize(x)[0])),
    [-15, 15]
);


# In[253]:


wfnet = WFNet(h2_plus, ion_pot=0.96, alpha=10)
plot_func_x(
    lambda x: gtowf(x)/wfnet._asymptote(wfnet._featurize(x)[0]),
    [-10, 10]
);


# In[42]:


1/2.6


# In[37]:


plot_func_x(
    gtowf,
    [-6, 6,]
);


# In[150]:


plot_func_xy(gtowf2, [[-3, 3], [-2, 2]]);


# In[33]:


integrate_on_mesh(lambda x: gtowf(x)**2, [(-6, 6), (-4, 4), (-4, 4)])


# In[10]:


plot_func_x(lambda x: local_energy(x, gtowf, h2_plus), [-3, 3])
plt.ylim((-10, 0));


# In[367]:


samples = HMC(
    dist=lambda x: wf_from_mf(x, mf, 0)**2,
    stepsize=np.sqrt(0.1),
    dysteps=0,
    n_walker=100,
    steps=5000,
    dim=3,
    startfactor=1,
    presteps=100,
).detach().transpose(0, 1).contiguous()


# In[61]:


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
info


# In[68]:


E_loc = local_energy(samples.view(-1, 3), gtowf, h2_plus).view(100, -1)


# In[69]:


plt.plot(*samples[0][:100, :2].numpy().T)
plt.gca().set_aspect(1)


# In[76]:


plt.hist2d(
    *samples.view(-1, 3)[:, :2].numpy().T,
    bins=100,
    range=[[-3, 3], [-3, 3]],
)                                   
plt.gca().set_aspect(1)


# In[75]:


plt.plot(E_loc.mean(0).numpy())
plt.ylim(-0.7, -0.5)


# In[77]:


E_loc.std()


# In[78]:


(E_loc.mean(dim=-1).std()*np.sqrt(5000)/E_loc.std())**2


# In[80]:


plt.hist(E_loc.flatten().clamp(-1.25, 0).numpy(), bins=100);


# In[48]:


scf_energy, E_loc.mean().item(), (E_loc.mean(dim=-1).std()/np.sqrt(100)).item()


# In[57]:


results = ([], [])
with open('/storage/mi/jhermann/Research/Projects/nqmc/_dev/test-calc-2/h2+-hf-vmc.qw.log') as f:
    for l in f:
        words = l.split()
        if words and words[0].startswith('total_energy'):
            i = 1 if words[0].endswith('var') else 0
            results[i].append(words[2])
results = torch.tensor(np.array(results, dtype=float).T)


# In[58]:


avg = results[:, 0].mean()
avg


# In[136]:


avgvar = (results[:, 1].view(-1, 1).norm(dim=-1)).mean().item()
np.sqrt(avgvar)


# In[137]:


(results[:, 0].var() + results[:, 1].mean()).sqrt().item()


# In[139]:


err = (results[:, 0].std(unbiased=False).item()/np.sqrt(500))**2
np.sqrt(err)


# In[140]:


indep_points = avgvar/err
indep_points


# In[68]:


plt.plot(*np.array(list(map(blocking, E_loc))).mean(0).T)


# In[69]:


plt.plot(np.array(list(map(lambda x: autocorr_coeff(range(100), x), E_loc))).mean(0))
plt.axhline()


# In[ ]:




