"""
@author: J.W.Spaak, jurg.spaak@unamur.be

Create communities with Holling type 1 response function and compute their
NFD values for different species richness
Ensure the all generated species can coexist in all communities and
no resource goes extinct"""

import numpy as np
import matplotlib.pyplot as plt
from nfd_definitions.numerical_NFD import NFD_model, InputError
from itertools import combinations

# global parameters
n_res = 2 # number of resources species share
n_spe_max = 5 # maximal number of species
n_com = 50 # number of communities to generate

# generater species parameters
# utilisation of resources by each species for common resources
util = np.random.uniform(1, 2, size = (n_spe_max, n_res, n_com))
# regeneration speed of each resource
regen = np.random.uniform(1, 2, size = (n_res, n_com))



# generate specialist resources
L_u_r = np.amax(util/regen, axis = 1)
B_u_r = (np.sum(util, axis = 1) - 
         np.sum(util**2/regen, axis = 1)/(2*n_spe_max * L_u_r))

u_div_r = L_u_r/2 * (1 + np.random.uniform(size = (n_spe_max, n_com)))
u_spec_diag = np.random.uniform(high = 0.1, size = (n_spe_max, n_com))
u_spec_diag = B_u_r * L_u_r* 2*n_spe_max/u_div_r*(1 + u_spec_diag)

u_spec = np.zeros((n_spe_max, n_spe_max, n_com))
u_spec[np.arange(n_spe_max), np.arange(n_spe_max)] = u_spec_diag

r_spec = u_spec_diag/u_div_r 

# combine specialist resources with general resources
util = np.append(u_spec, util, axis = 1)
regen = np.append(r_spec, regen, axis = 0)

# update boundary of mortality rate
L_u_r = np.amax(util/regen, axis = 1)
B_u_r = (np.sum(util, axis = 1) - 
         np.sum(util**2/regen, axis = 1)/(2*n_spe_max * L_u_r))

# generate mortality to ensure survival of species and resources
if (B_u_r>u_spec_diag).any():
    raise RuntimeError("No possible choices for mortality rate")
mort = np.random.uniform(size = (n_spe_max, n_com))
mort = B_u_r + mort*(u_spec_diag - B_u_r)

# compute LV model parameters assuming no resources go extinct
mu = np.sum(util, axis = 1) - mort
A = np.sum(util*util[:,np.newaxis]/regen, axis = 2)

class ResourceError(Exception):
    pass

def res_dens(N,u,r,m):
    """ growth of species with density N and trats u,r,m"""
    # resources density
    R_star = 1-np.einsum("s,sr->r", N, u)/r
    if (R_star<0).any(): # check whether no resource went extinct
        print(N)
        print(R_star)
        print(r)
        print(m)
        raise ResourceError("Negative resource density")
        
    # growth is resource consumption minus mortality
    return np.sum(u*R_star, axis = 1) - m

def LV(N,mu,A):
    # LV model, theoretically equivalent to res_dens
    return mu-A.dot(N)

# list of all possible combinations to select species
combs = []
for n_spec in range(2, n_spe_max+1):
    combs.extend(list(combinations(range(n_spe_max), n_spec)))
combs = [np.array(comb) for comb in combs]

# to save NFD values for the resource model
ND = [[] for i in range(n_spe_max + 1)]
FD = [[] for i in range(n_spe_max + 1)]

# to save NFD values for the LV model
ND_c = [[] for i in range(n_spe_max + 1)]
FD_c = [[] for i in range(n_spe_max + 1)]

for i in range(n_com):
    print(i) # progress report
    for comb in combs:
        n_spe = len(comb) # number of species
        # compute NFD values for LV model
        pars = NFD_model(LV, n_spec = n_spe, 
            args = (mu[comb,i], A[comb[:,None], comb,i]))
        ND_c[n_spe].append(pars["ND"].copy())   # save values
        FD_c[n_spe].append(pars["ND"].copy())
        
        # compute NFD values for the resource model
        # use pars of LV model as starting estimates
        pars = NFD_model(res_dens, n_spec = n_spe, 
            args = (util[comb,:,i], regen[:,i], mort[comb,i]),
                         pars = pars)
        ND[n_spe].append(pars["ND"])
        FD[n_spe].append(pars["ND"])
        
        if not np.isclose(ND[n_spe][-1], ND_c[n_spe][-1]).all():
            pars2 = NFD_model(LV, n_spec = n_spe, 
                             args = (mu[comb,i], A[comb[:,None], comb,i]))
            raise
        
        
# plotting and checking results       
ND = [np.array(nd) for nd in ND]
ND_c = [np.array(nd) for nd in ND_c]

# check whether the resutls are the same
check = [np.sum(~np.isclose(ND[i], ND_c[i]))/ND_c[i].size
         for i in range(len(ND))]
print(check)

plt.boxplot(ND)