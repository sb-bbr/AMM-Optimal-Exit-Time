from random import random
import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from scipy.integrate import solve_ivp

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt


class PathGenerator:
    
    def __init__(self, **params):

        np.random.seed(42)
        
        self.rate = params['rate']
        self.sigma = params['sigma']
        self.T = params['T']
        self.S0 = params['S0']

        self.n_steps = params['n_steps']
        self.n_paths = params['n_paths']
        
        self.a0 = params['a0']
        self.a1 = params['a1']
        self.a2 = params['a2']
        self.ksi = params['ksi']
        self.X0 = params['X0']
        self.Y0 = params['Y0']

        self.psi = params['psi']

        if 'c' not in params:
            self.c = self.X0 * self.Y0
        else:
            self.c = params['c'] #self.X0 * self.Y0

        self.phi_func = lambda x: self.c / x
        self.d_phi_func = lambda x: -self.c / (x ** 2)
        self.intensity_a = lambda y, S: np.maximum(self.a0, self.a1 + self.a2 * (S - (self.c / (y**2))))
        self.intensity_b = lambda y, S: np.maximum(self.a0, self.a1 + self.a2 * ((self.c / (y**2)) - S))
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
            'n_steps': self.n_steps,
            'n_paths': self.n_paths,
            'a0': self.a0,
            'a1': self.a1,
            'a2': self.a2,
            'ksi': self.ksi,
            'X0': self.X0,
            'Y0': self.Y0,
            'psi': self.psi
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

            R[:, i] = R[:, i - 1] + self.r_fees(Y[:, i - 1]) * (dNa_i + dNb_i)

        return X, Y, Z, R, S

    def amm_model(self, BM_type):
        
        X, Y, Z, R, S = self.__LP_paths(BM_type)

        self.paths['external_mid_price_S'] = S
        self.paths['amm_model_0'] = (X - self.X0) + S * (Y - self.Y0) + R
        #self.paths['amm_model_utility'] = -np.exp(-self.psi * ((X - self.X0) + S * (Y - self.Y0) + R))
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

    def longstaff_schwartz(self, paths, paths_S, paths_Y, deg, risk_neutral=True):
        
        n_steps = paths.shape[1] - 1
        n_paths = paths.shape[0]

        V = np.zeros_like(paths)
        V_psi = np.zeros_like(paths)

        tau_matrix = np.zeros_like(paths)
        tau_matrix[:, -1] = 1

        if risk_neutral:
            V[:, -1] = 0
        else:
            V[:, -1] = -np.exp(-self.psi*paths[:, -1])
            V_psi[:, -1] = 0 #-(1/self.psi)*np.log(-V[:, -1])-paths[:, -1]

        for i in range(n_steps - 1, 0, -1):
            
            try:
                
                Y_i = V[:, i + 1]
                
                if risk_neutral:

                    # Polynomial of state variables
                    X_i = np.column_stack((paths_S[:, i], paths_Y[:, i]))

                    poly = PolynomialFeatures(degree=deg, include_bias=False)
                    X_poly_i = poly.fit_transform(X_i)

                    model = LinearRegression()
                    model.fit(X_poly_i, Y_i)
                    continuation_value = model.predict(X_poly_i)

                    # PPD
                    exercise_value_i = 0
                    stop_here_i = exercise_value_i > continuation_value # Stop here when exercise value is greater than continuation value (>= means 'indifference' is accepted to stop)
                    V[:, i] = stop_here_i * exercise_value_i + (1 - stop_here_i) * (paths[:, i + 1] - paths[:, i] + V[:, i + 1])

                else:
                    # Polynomial of state variables
                    X_i = np.column_stack((paths_S[:, i], paths_Y[:, i], paths[:, i]))

                    poly = PolynomialFeatures(degree=deg, include_bias=True)
                    X_poly_i = poly.fit_transform(X_i)

                    model = LinearRegression(fit_intercept=False)
                    model.fit(X_poly_i, Y_i)
                    continuation_value = model.predict(X_poly_i)

                    # PPD
                    #exercise_value_i = -np.exp(-self.psi*(paths[:, i + 1]))
                    exercise_value_i = -np.exp(-self.psi*(paths[:, i]))
                    stop_here_i = exercise_value_i > continuation_value
                    V[:, i] =  stop_here_i * exercise_value_i + (1 - stop_here_i) * V[:, i + 1]
                    
                    # Transform back to original scale
                    V_psi[:, i] = -(1/self.psi)*np.log(-V[:, i])-paths[:, i]

                tau_matrix[:, i] = stop_here_i.astype(int)

            except Exception as e:
                print(f"Error at step {i}: {e}")

        stopping_time = np.argmax(tau_matrix, axis=1)

        V_matrix = V if risk_neutral else V_psi
        
        V0 = np.mean(V_matrix[:, 1], axis=0)

        return {'V0': V0, 'V_matrix': V_matrix, 'tau_matrix': tau_matrix, 'stopping_time': stopping_time}

    def euler(self, delta, h, jump_scale_nbr, S_scale_factor, risk_neutral=True):

        not_converge_counter = 0

        # Grid (time, space, jumps)
        time_discretization = np.arange(0, self.T + delta, delta)
        S_matrix = np.arange((1-S_scale_factor)*self.S0, (1+S_scale_factor)*self.S0 + h, h)
        y_matrix = np.arange(self.Y0 - jump_scale_nbr*self.ksi, self.Y0 + jump_scale_nbr*self.ksi + self.ksi, self.ksi)

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

        for i in tqdm(range(I - 1, -1, -1)): # From I - 1 to 0
            try:
                for l in range(0, L+1): # From 0 to L
                    
                    # Deal with jumps
                    V_temp = V_matrix[i + 1, l, :]

                    if L == 0: # No jumps
                        pass

                    else:
                        if l > 0:
                            # Adjust v(t_i, y_l, S_j) --> know v(t_i, y_l, S_j) for all l and j
                            if risk_neutral:
                                V_temp += delta*self.intensity_a(y_matrix[l], S_matrix)*(self.phi_func(y_matrix[l - 1]) - self.phi_func(y_matrix[l]) - self.ksi*S_matrix + self.r_fees(y_matrix[l]) + V_matrix[i + 1, l - 1, :] - V_matrix[i + 1, l, :])
                            else:
                                V_temp += delta*self.intensity_a(y_matrix[l], S_matrix)*(1/self.psi)*(1 - np.exp(-self.psi*(self.phi_func(y_matrix[l-1]) - self.phi_func(y_matrix[l]) - self.ksi*S_matrix + self.r_fees(y_matrix[l]) + V_matrix[i + 1, l-1, :] - V_matrix[i + 1, l, :])))

                        if l < L:
                            # Adjust v(t_i, y_l, S_j) --> know v(t_i, y_l, S_j) for all l and j
                            if risk_neutral:
                                V_temp += delta*self.intensity_b(y_matrix[l], S_matrix)*(self.phi_func(y_matrix[l + 1]) - self.phi_func(y_matrix[l]) + self.ksi*S_matrix + self.r_fees(y_matrix[l]) + V_matrix[i + 1, l + 1, :] - V_matrix[i + 1, l, :])
                            else:
                                V_temp += delta*self.intensity_b(y_matrix[l], S_matrix)*(1/self.psi)*(1 - np.exp(-self.psi*(self.phi_func(y_matrix[l + 1]) - self.phi_func(y_matrix[l]) + self.ksi*S_matrix + self.r_fees(y_matrix[l]) + V_matrix[i + 1, l + 1, :] - V_matrix[i + 1, l, :])))

                    ### Initial guess by solving a linear system (PDE without the square)
                    # Coefficient of the linear system
                    c_0_0, c_0_J = 1, 1
                    c_2 = 1 + delta * ((1/h**2)*self.sigma**2)
                    c_1_3 = -(1/(2*h**2)) * delta * (self.sigma**2)

                    # Define the linear system
                    A_c_2 = np.diag(np.concatenate((np.array([c_0_0]), np.ones(shape=(J-1,))*c_2, np.array([c_0_J]))), k=0) # diagonal
                    A_c_1 = np.diag(np.ones(shape=(J,))*c_1_3, k=1) # upper diagonal
                    A_c_3 = np.diag(np.ones(shape=(J,))*c_1_3, k=-1) # lower diagonal

                    A = A_c_1 + A_c_2 + A_c_3
                    A[0, 1] = 0
                    A[-1, -2] = 0

                    # Solve the linear system and use it as a the initial guess
                    V_newton = np.linalg.solve(A, V_temp)

                    ### Handle the non-linear part

                    if not risk_neutral:

                        # Solve the system --> know v(t_i, y_l, S_j) for all j
                        iters = 0
                        max_iters = 15
                        epsilon = 10**(-8)

                        F = lambda V: V*(1+delta*self.sigma**2/(h**2)) \
                                        + np.concatenate([V[1:], np.array([0])])*((-delta*self.sigma**2/(2*h**2) + (np.concatenate([V[1:], np.array([0])]) - np.concatenate([np.array([0]), V[:-1]]))*self.psi*delta*self.sigma**2/(4*h**2))) \
                                        + np.concatenate([np.array([0]), V[:-1]])*((-delta*self.sigma**2/(2*h**2) + (-np.concatenate([V[1:], np.array([0])]) + np.concatenate([np.array([0]), V[:-1]]))*self.psi*delta*self.sigma**2/(4*h**2)))
                    

                        V_newton = np.zeros(shape=(J+1,))

                        while np.sqrt(np.sum((F(V_newton) - V_temp)**2)) > epsilon and iters < max_iters:

                            # Neumann boundary conditions
                            c_0_0, c_0_J = 1, 1
                            
                            c_2 = 1 + delta * ((1/h**2)*self.sigma**2 - self.psi*self.sigma**2/(2*h))
                            c_1 = - (delta * (self.sigma**2) / (2*h**2)) + (self.psi * delta * self.sigma**2 / (4*h**2)) * (-V_newton[1:] + 2*V_newton[:-1])
                            c_3 = - (delta * (self.sigma**2) / (2*h**2)) + (self.psi * delta * self.sigma**2 / (4*h**2)) * (2*V_newton[:-1] - V_newton[1:])

                            J_c_2 = np.diag(np.concatenate((np.array([c_0_0]), np.ones(shape=(J-1,))*c_2, np.array([c_0_J]))), k=0) # diagonal
                            J_c_1 = np.diag(np.ones(shape=(J,))*c_1, k=-1) # lower diagonal
                            J_c_3 = np.diag(np.ones(shape=(J,))*c_3, k=1) # upper diagonal

                            Jacobian_matrix = J_c_2 + J_c_1 + J_c_3
                            Jacobian_matrix[0, 1] = 0
                            Jacobian_matrix[-1, -2] = 0

                            V_newton = V_newton - np.linalg.inv(Jacobian_matrix)@(F(V_newton) - V_temp)

                            iters+=1

                        if np.sqrt(np.sum((F(V_newton) - V_temp)**2)) > epsilon:
                            not_converge_counter+=1

                    V_matrix[i, l, :] = V_newton
                    
                    # Execution boundary, every point where - v(t_i, y_l, S_j) > 0 is set to 0 to satisfy the QVI
                    V_matrix[i, l, -V_matrix[i, l, :] > 0] = 0
                    V_matrix_QVI[i, l, :] = V_matrix[i, l, :] # Store the final value

                if not risk_neutral and not_converge_counter !=0:
                    print(f'Newton scheme did converge {not_converge_counter} over {L*I}')
            
            except Exception as e:
                print(f"Error at time step {i} and jump level {l}: {e}")

        return {'V_matrix':V_matrix_QVI, 'external_mid_price_S':S_matrix, 'jumps_grid':y_matrix}





### Old methods
'''
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
                                         + self.intensity_a(y_matrix[l], S_matrix)*(self.phi_func(y_matrix[l-1]) - self.phi_func(y_matrix[l]) - self.ksi*S_matrix + self.r_fees(y_matrix[l])))
                        if l < L:
                            # Adjust v(t_i, y_l, S_j) --> know v(t_i, y_l, S_j) for all l and j
                            
                            V_temp += delta*(V_matrix[i+1, l+1, :]*self.intensity_b(y_matrix[l], S_matrix) \
                                         + self.intensity_b(y_matrix[l], S_matrix)*(self.phi_func(y_matrix[l+1]) - self.phi_func(y_matrix[l]) + self.ksi*S_matrix + self.r_fees(y_matrix[l])))
                    

                    # Solve the linear system --> know v(t_i, y_l, S_j) for all j
                    V_matrix[i, l, :] = np.linalg.solve(A, V_temp)

                    # Execution boundary, every point where - v(t_i, y_l, S_j) > 0 is set to 0 to satisfy the QVI
                    V_matrix[i, l, -V_matrix[i, l, :] > 0] = 0
                    V_matrix_QVI[i, l, :] = V_matrix[i, l, :] # Store the final value

            except Exception as e:
                print(f"Error at time step {i} and jump level {l}: {e}")

        return {'V_matrix':V_matrix_QVI, 'external_mid_price_S':S_matrix, 'jumps_grid':y_matrix}
'''