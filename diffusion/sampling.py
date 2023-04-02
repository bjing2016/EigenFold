import tqdm, torch, math, copy
import numpy as np
from torch_geometric.transforms import BaseTransform


class ForwardDiffusionKernel(BaseTransform):
    def __init__(self, args):
        self.skew = args.train_skew
        self.Hf = args.train_Hf
        self.rmsd_max = args.train_rmsd_max
        self.tmin = args.train_tmin
        self.kmin = args.train_kmin
        self.cutoff = args.train_cutoff
        self.center = True
        self.cuda = args.cuda_diffuse
        self.key = 'resi'
        self.args = args
        
    def __call__(self, data):
        if data.skip: return data
        sde = data.sde; rsde = data.resi_sde
        
        step = np.random.rand()
        if step > 2/(1+np.exp(abs(self.skew))):
            step = np.random.beta(2, 1) if self.skew > 0 else np.random.beta(1, 2)
            
        rmsd_max = np.interp(self.Hf, rsde.hs[::-1], rsde.rmsds[::-1])
        if self.rmsd_max > 0:
            rmsd_max = min(rmsd_max, self.rmsd_max)
            
        rmsd_min = rsde.rmsd(self.tmin)
        rmsd = rmsd_max * step + rmsd_min * (1-step)
        
        t = np.interp(rmsd, rsde.rmsds, rsde.ts)
        
        k = (sde.D * t < self.cutoff).sum() - 1
        k = min(sde.N-1, max(self.kmin, k))
        
        data.step, data.t, data.k = step, t, k
        data.rmsd = rmsd
        
        # centering and SO(3) alignment...
        num_nodes = data[self.key].num_nodes
        data.score_norm = sde.score_norm(t, k, adj=True)
        data[self.key].node_t = torch.ones(num_nodes) * t
        
        pos = data[self.key].pos.numpy()
        
        pos, score = sde.sample(t, pos, center=self.center, score=True, k=k, adj=True)
        pos, score = torch.from_numpy(pos), torch.from_numpy(score)
        
        data[self.key].pos, data[self.key].score = pos, score
        data.score = score
        return data
            
def reverse_sample(args, score_func, sde, sched, pdb=None, Y=None, device='cpu', tqdm_=True, ode=False, logF=None):
    sde = copy.deepcopy(sde); 
    if 'cuda' in str(device): sde.cuda()
    
    if Y is None: Y = sde.sample(sched.tmax)
    else: Y = torch.tensor(Y).float().to(device)
    Y = sde.project(Y, sched.ks[0], center=False)
    
    if pdb: pdb.add(Y)
    
    steps = tqdm.trange(sched.N) if tqdm_ else range(sched.N)
    kold = sched.ks[0]
    for i in steps:

        t, dt, k, dk = sched.ts[i], sched.dt[i], sched.ks[i], sched.dk[i]
        k = (sde.D * t < args.train_cutoff).sum() - 1
        dk = k - kold; kold = k
        if dk:
            new_eigens = torch.zeros(sde.N, dtype=bool, device=device)
            new_eigens[k-dk+1:k+1] = True
            Y = Y + sde.inject(t, new_eigens)
            
        
        score = score_func(Y, t, k)
        
        dY = - (1 + sched.alpha*sched.beta) * (sde.P*sde.D)@(sde.P.T@Y) * dt / 2
        dY = dY + (1 + sched.alpha*sched.beta/2) * score * dt 
        
        if logF is not None:
            Y.grad = None; Y.requires_grad = True
            logF_ = logF(Y); logF_.sum().backward()
            dY = dY + sched.alpha/2 * Y.grad * dt
            Y.grad = None; Y.requires_grad = False
        
        dY = dY + np.sqrt(dt*(1+sched.alpha)) * torch.randn(*Y.shape, device=device)
        Y = Y + sde.project(dY, k, center=False)
        
        if pdb: pdb.add(Y)
    
    return Y.cpu().numpy()

@torch.no_grad()
def logp(Y, score_fn, sde, sched, pdb=None, device='cpu', tqdm_=True, seed=None):
    sde = copy.deepcopy(sde); sde.cuda()
    Y = torch.tensor(Y).float().to(device)
    ts, ks = sched.ts[::-1], sched.ks[::-1]
    
    elbo = 0
    
    iters = zip(ts[:-1], ts[1:], ks[:-1], ks[1:])
    if tqdm_: iters = tqdm.tqdm(iters, total=sched.N)
    for t0, t1, k0, k1 in iters:
        dt = t1-t0
        Y0 = Y
        
        logk_E, logk_Z = 0, 0
        eigens = sde.eigens(t0)
        dk = k0 - k1
        if k0 != k1:
            k_eigens = eigens[k1+1:k0+1]
            k_vec = (sde.P.T @ Y0)[k1+1:k0+1]
            logk_E = -(k_vec**2/k_eigens[:,None]).sum()/2
            logk_Z = -3*torch.log(k_eigens).sum()/2 
        
        mu1 = Y0 - sde.J @ Y0 / 2 * dt
        Y1 = mu1 + np.sqrt(dt) * torch.randn(*mu1.shape, device=device)
        Y1 = sde.project(Y1, k1, center=True)
        z1 = (Y1 - mu1)
        z1 = sde.project(z1, k1, center=True)
        logq_E = -(z1**2/dt).sum()/2 
        #logq_Z = -(k1)*3*np.log(dt)/2 
    
        mu0 = Y1 - (1/2) * sde.J @ Y1 * dt + score_fn(Y1, t1, k1) * dt
        z0 = Y0 - mu0
        z0 = sde.project(z0, k1, center=True)
        logp_E = -(z0**2/dt).sum()/2 
        #logp_Z = -(k1)*3*np.log(dt)/2 
        
        elbo = elbo + logk_E + logk_Z + logp_E - logq_E
        Y = Y1

    logP_E = -((sde.P.T @ Y)[1:k1+1]**2*sde.D[1:k1+1,None]).sum()/2 
    logP_Z = 3*torch.log(sde.D[1:k1+1]).sum()/2
    
    elbo = elbo + logP_E + logP_Z - 3*(sde.N-1)/2 * np.log(2*np.pi)
    dof = 3*sde.N - 3
    return float(elbo/dof)