from scipy.optimize import root_scalar, minimize
import numpy as np

class Schedule:
    def __init__(self, sde, Hf=0.1, step=0.1, rmsd_max=0., cutoff=10, kmin=5, tmin=0.2, alpha=0, beta=1):
        
        self.D = D = sde.D[1:]
        tmax = root_scalar(lambda t: sde.KL_H(t)-Hf, bracket=[tmin, 1e8]).root
        if sde.rmsd(tmax) > rmsd_max > 0:
            tmax = root_scalar(lambda t: sde.rmsd(t)-rmsd_max, bracket=[tmin, tmax]).root
        
        self.ts = ts = [tmax]
        kmin = min(len(self.D), kmin)
        self.alpha, self.beta = alpha, beta
        self.populate(step, tmin, cutoff, kmin, alpha, beta)
        
        ks = [(cutoff / D > t).sum() for t in ts]
        self.ks = np.array(np.maximum(kmin, ks))
        
        self.cutoff = np.zeros_like(self.D)
        for i, dd in enumerate(D):
            steps_without = (self.ks <= i).sum()
            self.cutoff[i] = self.ts[steps_without]
            
        self.ts = np.array(ts); self.tmax = ts[0]; self.tmin = ts[-1]
        self.dt = -np.diff(ts)
        self.dk = np.array([0] + list(np.diff(self.ks)))
        self.N = len(ts)-1
    
    def KL_H(self, skip=0):
        skip_t = self.ts[skip]
        cutoff = np.minimum(self.cutoff, skip_t)
        return -3*0.5*(np.log(1-np.exp(-self.D*cutoff)) + np.exp(-self.D*cutoff))
    
    def KL_E(self, E):
        return np.exp(-self.D*self.cutoff)*E

class EntropySchedule(Schedule):
    def populate(self, dH, tmin, cutoff, kmin, alpha, beta):
        ts = self.ts
        while ts[-1] > tmin:
            tnext = -(1/self.D)*np.log(((np.exp(dH)-1)+np.exp(-self.D*ts[-1]))/np.exp(dH))
            ts.append(tnext.max())
    
class RateSchedule(Schedule):    
    def populate(self, step, tmin, cutoff, kmin, alpha, beta):
        ts = self.ts
        while ts[-1] > tmin:
            dt = step*(1-np.exp(-self.D*ts[-1]))/self.D
            dt = dt/(1 + alpha*beta/2)
            k = max(kmin, (self.D*ts[-1] < cutoff).sum())
            ts.append(ts[-1]-dt[:k].min())
            
            
