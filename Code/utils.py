from random import random
import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from scipy.integrate import solve_ivp


class environment:
    """
    Model parameters for the environment.
    """
    def __init__(self, sigma = 1., S0 = 100., Z0 = 100., Y0 = 1000., eta = 0.1, xi = 1., T = 1., 
                 a1 = 1, a2 = 1, Nt =1_000, r = 0.5):
        self.sigma = sigma
        self.T = T
        self.S0 = S0
        self.Z0 = Z0
        self.Y0 = Y0
        self.eta = eta
        self.Nt = Nt
        self.a1 = a1
        self.a2 = a2
        self.xi = xi
        self.r = r
        self.timesteps = np.linspace(0, self.T, num = (Nt+1))
        self.dt = self.T/self.Nt

    def simulate_price_market(self, nsims=1):
        x = np.zeros((self.Nt+1, nsims))
        x[0,:] = self.S0
        errs = np.random.randn(self.Nt, nsims)
        sigma = self.sigma
        for t in range(self.Nt):
            x[t + 1,:] = x[t,:] + np.sqrt(self.dt) * sigma * errs[t,:]
        return x
    
    def lambda_a(self, Zt, St):
        Zt = Zt.reshape((-1,))
        St = St.reshape((-1,))
        nsims = len(Zt)
        zeros_n = np.zeros((nsims,))
        lambdaa = np.maximum(0, self.a1 + self.a2*(St-Zt))*self.dt
        return lambdaa
    
    def lambda_b(self, Zt, St):
        Zt = Zt.reshape((-1,))
        St = St.reshape((-1,))
        nsims = len(Zt)
        zeros_n = np.zeros((nsims,))
        lambdab = np.maximum(0, self.a1 + self.a2*(Zt-St))*self.dt
        return lambdab
    
    def simulate_jumps(self, Zt, St):
        Zt = Zt.reshape((-1,))
        St = St.reshape((-1,))
        nsims = len(Zt)
        zeros_n = np.zeros((nsims,))
        lambdaa = self.lambda_a(Zt, St)
        lambdab = self.lambda_b(Zt, St)
        jump_a = (np.random.uniform(size=(nsims,)) < lambdaa).astype(int)
        jump_b = (np.random.uniform(size=(nsims,)) < lambdab).astype(int)
        return jump_a, jump_b, lambdaa, lambdab


    
    
    
    def solve_v123456(self):
        _ts = self.timesteps
        _Gt = lambda t, v: -np.array([
                                        #ODE associated with  v[1] 
                                        +self.sigma*v[5] - 2.*self.a1 * self.eta**2 * v[4],
                                        #ODE associated with  v[2] 
                                        2*self.a2*self.eta* v[1],
                                        #ODE associated with  v[3] 
                                        -2*self.a2*self.eta* v[1] + 2* self.a1*self.r*self.xi,
                                        #ODE associated with  v[4] 
                                        -4*self.a2*self.eta* v[4] + 2* self.a2*self.eta*v[3] - 4*self.a2*self.xi,
                                        #ODE associated with  v[5] 
                                        4*self.a2*self.eta* v[4] + 2* self.a2*self.xi,
                                        #ODE associated with  v[6] 
                                        -2*self.a2*self.eta* v[3] + 2* self.a2*self.xi
                                        ] )
        _sol        = solve_ivp(_Gt, 
                            [self.T, 0], 
                            np.array([0, 0, 0, 0, 0, 0]), 
                            t_eval = _ts[::-1])
        _Gt         = _sol.y
        return _Gt