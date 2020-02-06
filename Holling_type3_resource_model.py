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
n_com = 20 # number of communities to generate
K = 1 # carrying capacity, asumed to be one for all resources
holl_exp = 2

# generater species parameters
# utilisation of resources by each species for common resources
util = np.random.uniform(1, 2, size = (n_spe_max, n_res, n_com))
# regeneration speed of each resource
regen = np.random.uniform(1, 2, size = (n_res, n_com))
# halfsaturation constant
H_sat = np.ones((n_spe_max, n_res, n_com))



H_spec_diag = np.ones((n_spe_max, n_com))
H_spec = np.ones((n_spe_max, n_spe_max, n_com))
H_spec[np.arange(n_spe_max), np.arange(n_spe_max)] = H_spec_diag

# minimal boundary for utilisation of specific resource
u_spec_diag = 1
# increase utilisation on average by x%
u_spec_diag *= np.random.uniform(1, 1.1, size = (n_spe_max, n_com))

u_spec = np.zeros((n_spe_max, n_spe_max, n_com))
u_spec[np.arange(n_spe_max), np.arange(n_spe_max)] = u_spec_diag


r_spec = u_spec_diag/H_spec_diag

# combine specialist resources with general resources
util = np.append(u_spec, util, axis = 1)
regen = np.append(r_spec, regen, axis = 0)
H_sat = np.append(H_spec, H_sat, axis = 1)

n_res = util.shape[1] # added specific resources
mort = np.random.uniform(size = (n_spe_max, n_com))
mort *= u_spec_diag*K**holl_exp/(K**holl_exp+H_spec_diag**holl_exp)




# compute LV model parameters assuming no resources go extinct
mu = np.sum(util/H_sat, axis = 1) - mort
A = np.sum(util/H_sat*(util/H_sat)[:,np.newaxis]/regen, axis = 2)

class ResourceError(Exception):
    pass

def res_dens(N, u, r, H, m):
    """ growth of species with density N and trats u,r,m"""
    # resources density
    try:
        R_star = solver_jurg(holling_type3, 1.1*K*np.ones(n_res),
                         np.zeros(n_res),args = (N, r, u, H))
    except ValueError: # resources are not surviving
        return -m # assume all resources are extinct
    
    # growth is resource consumption minus mortality
    return np.sum(u*R_star/(H**holl_exp+R_star**holl_exp), axis = 1) - m

def solver_jurg(f,a,b, args = (), rel_tol = 1e-8, abs_tol = 1e-12):
    a,b = a.copy(),b.copy()
    if not ((f(a,*args)<0).all() and (f(b,*args)>0).all()):
        raise ValueError("JSP,f(a) must be negative and f(b) must be positive")
    rel_diff, abs_diff = np.inf, np.inf
    counter = 0
    while rel_diff > rel_tol and abs_diff > abs_tol :
        change_b = f((a+b)/2,*args)>0
        b[change_b] = ((a+b)/2)[change_b]
        a[~change_b] = ((a+b)/2)[~change_b]
        rel_diff = np.amax(np.abs(b-a)/a)
        abs_diff = np.amax(np.abs(b-a))
        counter +=1
    return (a+b)/2

def holling_type3(R, N, r, u, H):
    return r*(1-R/K) - np.sum(N.reshape(-1,1)*u/(R**holl_exp+H**holl_exp),
              axis = 0)

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
        
        # compute NFD values for the resource model
        # use pars of LV model as starting estimates
        # pars["N_star"] /= 5
        try:
            pars = NFD_model(res_dens, n_spec = n_spe, 
                args = (util[comb,:,i], regen[:,i],H_sat[comb, :, i],
                        mort[comb,i]), xtol = 1e-3,
                         pars = {"N_star": 1e-5*np.ones((n_spe, n_spe))})
            ND[n_spe].append(pars["ND"])
            FD[n_spe].append(pars["FD"])
        except InputError:
            print(i, comb, "not found")
        
        
        
        
# plotting results

fig = plt.figure(figsize = (10,10))


ax_FD = fig.add_subplot(2,2,3)
ax_ND = fig.add_subplot(2,2,1)

ax_ND.boxplot(ND[2:], positions = range(2,n_spe_max+1))
ax_FD.boxplot(FD[2:], positions = range(2,n_spe_max+1))
ax_coex = fig.add_subplot(1,2,2)

for i in range(2,n_spe_max+1):
    ax_coex.scatter(ND[i], -np.array(FD[i]))      

y_lim = ax_coex.get_ylim()
ND_bound = np.linspace(-2,2,101)
ax_coex.plot(ND_bound, ND_bound/(1-ND_bound), "black")
ax_coex.set_xlim(0,1)
ax_coex.set_ylim(-np.array(ax_FD.get_ylim())[::-1])
# add layout
ax_ND.set_title("Lotka Volterra")

ax_FD.set_xlabel("species richness")
ax_FD.set_ylabel("FD")
ax_ND.set_ylabel("ND")

ax_coex.set_ylabel("-FD")
ax_coex.set_xlabel("ND")

fig.tight_layout()

fig.savefig("Holling_type2_resource_explicit.png")