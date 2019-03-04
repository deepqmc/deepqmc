import numpy as np
import torch
from tqdm import tqdm, trange


def dynamics(dist, pos, stepsize, steps):

    pos = pos.detach().clone()
    pos.requires_grad = True
    vel = torch.randn(pos.shape)
    v_te2 = (
        vel
        - stepsize
        * torch.autograd.grad(
            dist(pos),
            pos,
            create_graph=True,
            retain_graph=True,
            grad_outputs=torch.ones(pos.shape[0]),
        )[0]
        / 2
    )
    p_te = pos + stepsize * v_te2
    for _ in range(1, steps):
        v_te2 = (
            v_te2
            - stepsize
            * torch.autograd.grad(
                dist(p_te),
                p_te,
                create_graph=True,
                retain_graph=True,
                grad_outputs=torch.ones(pos.shape[0]),
            )[0]
        )
        p_te = p_te + stepsize * v_te2
        
    v_te = (
        v_te2 
        - stepsize / 2 
        * torch.autograd.grad(
            dist(pos),
            pos,
            create_graph=True,
            retain_graph=True,
            grad_outputs=torch.ones(pos.shape[0]),
        )[0])

    return p_te, v_te, vel


def HMC(
    dist,
    stepsize,
    dysteps,
    n_walker,
    steps,
    dim,
    startfactor=1,
    presteps=200,
):

    acc = 0
    samples = torch.zeros(steps, n_walker, dim)
    walker = torch.randn(n_walker, dim) * startfactor
    distwalker = dist(walker).detach().numpy()

    for i in trange(steps + presteps):

        if i >= presteps:
            samples[i - presteps] = walker

        trial, v_trial, v_0 = dynamics((lambda x: -torch.log(dist(x))), walker, stepsize, dysteps)
        disttrial = dist(trial).detach().numpy()
        ratio = torch.from_numpy(disttrial / distwalker) * (
            torch.exp(
                -0.5
                * (torch.sum(v_trial ** 2, dim=-1) - torch.sum(v_0 ** 2, dim=-1))
            )
        )
        R = torch.rand(n_walker)
        smaller = (ratio < R).type(torch.LongTensor)
        larger = torch.abs(smaller - 1)
        ind = torch.nonzero(larger).flatten()
        walker[ind] = trial[ind]
        distwalker[ind] = disttrial[ind]
        if i >= presteps:
            acc += torch.sum(larger).item()

    print('Acceptanceratio: ' + str(np.round(acc / (n_walker * steps) * 100, 2)) + '%')
    return samples


def HMC_ad(
    dist,
    stepsize,
    dysteps,
    n_walker,
    steps,
    dim,
    push,
    startfactor=1,
    T=1,
    presteps=200,
):

    acc = 0
    ac = 0.8
    samples = torch.zeros(steps, n_walker, dim)
    walker = torch.randn(n_walker, dim) * startfactor
    #v_walker = torch.zeros((n_walker, dim))
    distwalker = dist(walker)
    disttrial = torch.zeros(n_walker)

    for i in trange(steps + presteps):

        if i >= presteps:
            samples[i - presteps] = walker
        trial, v_trial, v_0 = dynamics((lambda x: -dist(x)), walker, stepsize, dysteps, push)
        trial = 2 * trial - walker
        v_trial = v_trial

        na = torch.linspace(0, n_walker - 1, n_walker).type(torch.LongTensor)
        count = 0
        smaller = torch.zeros(n_walker).type(torch.LongTensor)
        larger = torch.zeros(n_walker).type(torch.LongTensor)
        ratio = torch.zeros(n_walker).type(torch.FloatTensor)
        R = torch.rand(len(na))

        while (len(na) / n_walker) > (1 - ac) and count < 10:
            count += 1
            trial[na] = (trial[na] + walker[na]) / 2
            # v_trial[na] = v_trial[na]/2
            disttrial = dist(trial[na])
            ratio[na] = (disttrial / distwalker[na]) * (
                torch.exp(
                    -0.5
                    * (
                        torch.sum(v_trial[na] ** 2, dim=-1)
                        - torch.sum(v_0[na] ** 2, dim=-1)
                    )
                )
            )
            smaller[na] = (ratio[na] < R[na]).type(torch.LongTensor)
            larger[na] = torch.abs(smaller[na] - 1)
            ind = torch.nonzero(larger[na]).flatten()
            walker[na[ind]] = trial[na[ind]]
            #v_walker[na[ind]] = v_trial[na[ind]]
            distwalker[na[ind]] = disttrial[ind]
            if i >= presteps:
                acc += torch.sum(larger[na]).item()
            na = torch.nonzero(smaller[na]).flatten()
    print('Acceptanceratio: ' + str(np.round(acc / (n_walker * steps) * 100, 2)) + '%')
    return samples


def metropolis(
    distribution,
    startpoint,
    stepsize,
    steps,
    dim,
    n_walker,
    startfactor=0.2,
    presteps=0,
    interval=None,
    T=0.2,
):
    # initialise list to store walker positions
    samples = torch.zeros(steps, n_walker, len(startpoint))
    # another list for the ratios
    ratios = np.zeros((steps, n_walker))
    # initialise the walker at the startposition
    walker = torch.randn(n_walker, dim) * startfactor
    # plt.plot(walker.detach().numpy()[:,0],walker.detach().numpy()[:,1], \
    #          marker='.',ms='1',ls='',color='k')
    distwalker = distribution(walker)
    # loop over proposal steps
    for i in trange(presteps + steps):
        # append position of walker to the sample list in case presteps exceeded
        if i > (presteps - 1):
            samples[i - presteps] = walker
            # propose new trial position
            # trial = walker + (torch.rand(6)-0.5)*maxstepsize
            # pro = torch.zeros(2)
        pro = (torch.rand(walker.shape) - 0.5) * stepsize
        trial = walker + pro
        # calculate acceptance propability
        disttrial = distribution(trial)
        # check if in interval
        if interval is not None:
            inint = torch.tensor(
                all(torch.tensor(interval[0]).type(torch.FloatTensor) < trial[0])
                and all(torch.tensor(interval[1]).type(torch.FloatTensor) > trial[0])
            ).type(torch.FloatTensor)
            disttrial = disttrial * inint

        ratio = np.exp((disttrial.detach().numpy() - distwalker.detach().numpy()) / T)
        ratios[i - presteps] = ratio
        # accept trial position with respective propability
        smaller_n = (ratio < np.random.uniform(0, 1, n_walker)).astype(float)
        # larger_n = np.abs(smaller_n - 1)
        smaller = torch.from_numpy(smaller_n).type(torch.FloatTensor)
        larger = torch.abs(smaller - 1)
        walker = trial * larger[:, None] + walker * smaller[:, None]

        # if ratio > np.random.uniform(0,1):
        # 		walker = trial
        # 		distwalker = disttrial
        # return list of samples
        # print("variance of acc-ratios = \
        #   " + str((np.sqrt(np.mean(ratios**2)-np.mean(ratios)**2)).data))
    return samples
