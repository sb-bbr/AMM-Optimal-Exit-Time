from random import random
import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from scipy.integrate import solve_ivp


class PathGenerator:
    
    def __init__(self, **params): 
        
        self.rate = params['rate']
        self.sigma = params['sigma']
        self.T = params['T']
        self.S0 = params['S0']
        self.strike = params['strike']

        self.n_steps = params['n_steps']
        self.n_paths = params['n_paths']
        
        self.a0 = params['a0']
        self.a1 = params['a1']
        self.a2 = params['a2']
        self.ksi = params['ksi']
        self.X0 = params['X0']
        self.Y0 = params['Y0']

        self.gamma = params['gamma']

        self.c = self.X0 * self.Y0

        self.phi_func = lambda x: self.c / x
        self.d_phi_func = lambda x: -self.c / (x ** 2)
        self.intensity_a = lambda y, S: np.maximum(self.a0, self.a1 + self.a2 * (S - self.c / (y**2)))
        self.intensity_b = lambda y, S: np.maximum(self.a0, self.a1 + self.a2 * (self.c / (y**2) - S))
        self.r_fees = lambda Z: params['fees_coeff']*Z 
        
        self.paths = {}

    def get_params(self):
        """
        Retrieve the parameters of the model.
        """
        return {
            'rate': self.rate,
            'sigma': self.sigma,
            'T': self.T,
            'S0': self.S0,
            'strike': self.strike,
            'n_steps': self.n_steps,
            'n_paths': self.n_paths,
            'a0': self.a0,
            'a1': self.a1,
            'a2': self.a2,
            'ksi': self.ksi,
            'X0': self.X0,
            'Y0': self.Y0,
            'gamma': self.gamma
        }
    
    def external_mid_price_paths(self, BM_type):
        """
        Generate the Brownian motion paths based on the parameters.
        """
        Z = np.random.normal(0, 1, size=(self.n_paths, self.n_steps))
        dt = self.T / self.n_steps
        
        if BM_type == 'geometric':
            dX = np.zeros((self.n_paths, self.n_steps + 1))
            dX[:, 0] = np.log(self.S0)
            dX[:, 1:] = (self.rate - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt)*Z       
            
            X = np.exp(np.cumsum(dX, axis=1))
        
        elif BM_type == 'arithmetic':
            dX = np.zeros((self.n_paths, self.n_steps + 1))
            dX[:, 0] = self.S0
            dX[:, 1:] = self.sigma * np.sqrt(dt) * Z

            X = np.cumsum(dX, axis=1)
        
        else:
            raise ValueError("BM_type must be either 'geometric' or 'arithmetic'.")

        return X

    def __LP_paths(self, BM_type):
        
        step_size = self.T / self.n_steps

        S = self.external_mid_price_paths(BM_type)
        X = np.zeros_like(S)
        Y = np.zeros_like(S)
        Z = np.zeros_like(S)
        R = np.zeros_like(S)
        Na = np.zeros_like(S)
        Nb = np.zeros_like(S)

        Y[:, 0] = self.Y0
        X[:, 0] = self.X0
        Z[:, 0] = X[:, 0] / Y[:, 0]

        for i in range(1, self.n_steps + 1):
            lambda_a = self.intensity_a(Y[:, i - 1], S[:, i])
            lambda_b = self.intensity_b(Y[:, i - 1], S[:, i])

            u_a = np.random.uniform(0, 1, size=self.n_paths)
            u_b = np.random.uniform(0, 1, size=self.n_paths)

            dNa_i = (u_a <= lambda_a * step_size).astype(int)
            dNb_i = (u_b <= lambda_b * step_size).astype(int)

            Na[:, i] = dNa_i
            Nb[:, i] = dNb_i

            Y[:, i] = Y[:, i - 1] + self.ksi * dNb_i - self.ksi * dNa_i
            X[:, i] = X[:, i - 1] + (self.phi_func(Y[:, i - 1] + self.ksi) - self.phi_func(Y[:, i - 1])) * dNb_i \
                                    + (self.phi_func(Y[:, i - 1] - self.ksi) - self.phi_func(Y[:, i - 1])) * dNa_i
            Z[:, i] = Z[:, i - 1] + (-self.d_phi_func(Y[:, i - 1] + self.ksi) + self.d_phi_func(Y[:, i - 1])) * dNb_i \
                                    + (-self.d_phi_func(Y[:, i - 1] - self.ksi) + self.d_phi_func(Y[:, i - 1])) * dNa_i

            R[:, i] = R[:, i - 1] + self.r_fees(self.c / (Y[:, i - 1]**2)) * (dNa_i + dNb_i)

        return X, Y, Z, R, S

    def amm_model(self, BM_type):
        
        X, Y, Z, R, S = self.__LP_paths(BM_type)

        self.paths['external_mid_price_S'] = S
        self.paths['amm_model_0'] = (X - self.X0) + S * (Y - self.Y0) + R
        self.paths['amm_model_utility'] = np.exp(-self.gamma * ((X - self.X0) + S * (Y - self.Y0) + R))
        self.paths['asset_X'] = X
        self.paths['asset_Y'] = Y
        self.paths['asset_Z'] = Z
        self.paths['fees'] = R

    def get_paths(self):
        """
        Retrieve the generated paths by name.
        """
        return self.paths

class Solver(PathGenerator):

    def __init__(self, **params):
        
        super().__init__(**params)

    def longstaff_schwartz(self, paths, deg, problem_type):
        
        ITM, ITM_bool = None, False

        discount_factor = np.exp(-self.rate * (self.T / self.n_steps))
        
        n_steps = paths.shape[1] - 1
        n_paths = paths.shape[0]

        if problem_type == 'call':
            payoff_func = lambda x, K: np.maximum(x - K, 0)
            terminal_condition = payoff_func(paths[:, -1], self.strike)
        elif problem_type == 'put':
            payoff_func = lambda x, K: np.maximum(K - x, 0)
            terminal_condition = payoff_func(paths[:, -1], self.strike)
        elif problem_type == 'amm':
            payoff_func = lambda x, K: x
            terminal_condition = 0
            ITM = np.ones(n_paths, dtype=bool)
        else:
            raise ValueError("Problem_type must be either 'call', 'put', or 'amm'.")

        V = np.zeros_like(paths)
        V[:, -1] = terminal_condition

        tau_matrix = np.zeros_like(paths)
        tau_matrix[:, -1] = 1

        if ITM is None: ITM_bool = True
        
        for i in range(n_steps - 1, 0, -1):
            try:
                if ITM_bool:
                    ITM = payoff_func(paths[:, i], self.strike) > 0

                if np.sum(ITM) > 0:

                    X_i = paths[ITM, i]
                    Y_i = V[ITM, i + 1] * discount_factor
                    coeffs = np.polynomial.polynomial.polyfit(X_i, Y_i, deg=deg)

                    continuation_value = np.polynomial.polynomial.polyval(X_i, coeffs)
                    exercise_value_i = payoff_func(X_i, self.strike)

                    stop_here_i = exercise_value_i >= continuation_value # Stop here when exercise value is greater than continuation value (>= means 'indifference' is accepted to stop)
                    
                    V[ITM, i] = stop_here_i * exercise_value_i + (1 - stop_here_i) * V[ITM, i + 1] * discount_factor

                    tau_matrix[ITM, i] = stop_here_i.astype(int)

                V[~ITM, i] = V[~ITM, i + 1] * discount_factor

            except Exception as e:
                print(f"Error at step {i}: {e}")

        stopping_time = np.argmax(tau_matrix, axis=1)
        V0 = np.mean(V[:, 1], axis=0) * discount_factor
        #print(V[:,1])

        return {'V0': V0, 'V_matrix': V, 'tau_matrix': tau_matrix, 'stopping_time': stopping_time}

    def euler(self, delta, h, jump_scale_nbr, S_scale_factor):

        # Grid (time, space, jumps)
        time_discretization = np.arange(0, self.T + delta, delta)
        S_matrix = np.arange((1-S_scale_factor)*self.S0, (1+S_scale_factor)*self.S0 + h, h)
        y_matrix = np.arange(self.Y0 - jump_scale_nbr*self.ksi, self.Y0 + jump_scale_nbr*self.ksi + self.ksi, self.ksi)


        # Coefficient of the linear system
        c_0_func = lambda y, S : 1 + delta * (self.intensity_a(y,S) + self.intensity_b(y,S))
        c_2_func = lambda y, S : 1 + delta * ((1/h**2)*self.sigma**2 + self.intensity_a(y,S) + self.intensity_b(y,S))
        c_1 = -(1/(2*h**2)) * delta * (self.sigma**2)

        # Backward scheme

        I = time_discretization.shape[0] - 1
        L = y_matrix.shape[0] - 1
        J = S_matrix.shape[0] - 1

        V_matrix = np.zeros(shape=(I + 1, L + 1, J + 1))

        # Terminal condition
        V_matrix[-1, :, :] = 0 #np.maximum(0, S_matrix - self.strike) # Call option payoff

        # To store the final value of the option
        V_matrix_QVI = np.zeros_like(V_matrix)
        V_matrix_QVI[-1, :, :] = 0 #np.maximum(0, S_matrix - self.strike) # Call option payoff

        for i in range(I - 1, -1, -1): # From I - 1 to 0
            try:
                for l in range(0, L+1): # From 0 to L

                    # Neumann boundary conditions: second derivative is 0   
                    c_0_0 = c_0_func(y_matrix[l], S_matrix[0]) # v(t_{i+1}, y_l, S_0)
                    c_0_J = c_0_func(y_matrix[l], S_matrix[-1]) # v(t_{i+1}, y_l, S_J})
                    
                    c_2 = c_2_func(y_matrix[l], S_matrix[1:-1]) # v(t_{i+1}, y_l, S_1), ..., v(t_{i+1}, y_l, S_J-1)

                    # Define the linear system
                    A_c_2 = np.diag(np.concatenate((np.array([c_0_0]), c_2, np.array([c_0_J]))), k=0) # diagonal
                    A_c_1 = np.diag(np.ones(shape=(J,))*c_1, k=1) # upper diagonal
                    A_c_3 = np.diag(np.ones(shape=(J,))*c_1, k=-1) # lower diagonal

                    A = A_c_1 + A_c_2 + A_c_3
                    A[0, 1] = 0
                    A[-1, -2] = 0
                    
                    # Deal with jumps
                    V_temp = V_matrix[i+1, l, :]

                    if L == 0: # No jumps
                        pass

                    else:
                        if l > 0:
                            # Adjust v(t_i, y_l, S_j) --> know v(t_i, y_l, S_j) for all l and j
                            V_temp += delta*(V_matrix[i+1, l-1, :]*self.intensity_a(y_matrix[l], S_matrix) \
                                        + self.intensity_a(y_matrix[l], S_matrix)*(self.phi_func(y_matrix[l-1]) - self.phi_func(y_matrix[l]) - self.ksi*S_matrix + self.r_fees(self.c/(y_matrix[l]**2))))
                        if l < L:
                            # Adjust v(t_i, y_l, S_j) --> know v(t_i, y_l, S_j) for all l and j
                            V_temp += delta*(V_matrix[i+1, l+1, :]*self.intensity_b(y_matrix[l], S_matrix) \
                                        + self.intensity_b(y_matrix[l], S_matrix)*(self.phi_func(y_matrix[l+1]) - self.phi_func(y_matrix[l]) + self.ksi*S_matrix + self.r_fees(self.c/(y_matrix[l]**2))))
                    
                    # Solve the linear system --> know v(t_i, y_l, S_j) for all j
                    V_matrix[i, l, :] = np.linalg.solve(A, V_temp)

                    # Execution boundary, every point where - v(t_i, y_l, S_j) > 0 is set to 0 to satisfy the QVI
                    V_matrix[i, l, -V_matrix[i, l, :] > 0] = 0
                    V_matrix_QVI[i, l, :] = V_matrix[i, l, :] # Store the final value

            except Exception as e:
                print(f"Error at time step {i} and jump level {l}: {e}")

        return {'V_matrix':V_matrix_QVI, 'external_mid_price_S':S_matrix, 'jumps_grid':y_matrix}


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
                                        +2*self.a2*self.eta* v[1],
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