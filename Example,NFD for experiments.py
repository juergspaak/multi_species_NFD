"""
@author: J.W.Spaak
Example how to compute the ND and FD for a given differential equation setting
"""

import numpy as np
from numerical_NFD import NFD_model
from NFD_for_experiments import NFD_experiment
from scipy.integrate import odeint

# create the differential equation system
n_spec = 2 # number of species in the system
np.random.seed(5) # set random seed for reproduce ability

# Lotka-Volterra model
A = np.random.uniform(0.5,1,(n_spec,n_spec)) # interaction matrix
np.fill_diagonal(A,np.random.uniform(1,2,n_spec)) # to ensure coexistence
mu = np.random.uniform(1,2,n_spec) # intrinsic growth rate
def test_f(N,t = 0):
    return mu - np.dot(A,N)

# create dataset for monoculture
time_exp1 = time_exp2 = np.linspace(0,10,20)
dens_exp1, dens_exp2 = np.empty((2,2,len(time_exp1)))

dens_exp1[0] = odeint(lambda N,t: N*test_f(N),[1e-3,0],time_exp1)[:,0]
dens_exp1[1] = odeint(lambda N,t: N*test_f(N),[0,1e-3],time_exp1)[:,1]

dens_exp2[0] = odeint(lambda N,t: N*test_f(N),[2*dens_exp1[0,-1],0]
                            ,time_exp1)[:,0]
dens_exp2[1] = odeint(lambda N,t: N*test_f(N),[0,2*dens_exp1[1,-1]]
                            ,time_exp1)[:,1]
N_star = mu/np.diag(A)
r_i = mu-N_star[::-1]*A[[0,1],[1,0]]

res = NFD_experiment(N_star, time_exp1, dens_exp1, time_exp2, dens_exp2,
                     r_i,  visualize = True)

pars_model = NFD_model(test_f)