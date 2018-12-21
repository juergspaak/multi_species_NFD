import numpy as np
import model_holling_discrete_res as mh
from nfd_definitions.numerical_NFD import NFD_model, InputError

import sys
from timeit import default_timer as timer
start = timer()
# n_spec = int(sys.argv[1])
n_com = 10

# resources
n_res = 10 #max(n_spec) # number of resources
id_res = np.arange(n_res) # identity of resources

K = np.ones(n_res) # carrying capacity of the resources
r = 2*np.ones(n_res) # regeneration speed of the resources

# to store NO and FD, for the 3 different models
NO_all, FD_all = np.full((2,3,n_com,n_spec), np.nan)

def LV_model(N,mu,A):
    return mu-A.dot(N)

def new_community(n_com, n_spec):
    # create new traits for a community
    u = 1/(2*n_res)
    util = u*2**np.random.uniform(-1,1,size = (n_com, n_spec, n_res))
    util[:,np.arange(n_spec), np.arange(n_spec)] = 1
    
    H = 2*2**np.random.uniform(-1,1,(n_com,n_spec, n_res))
    H[:,np.arange(n_spec), np.arange(n_spec)] = 1
    
    max_m = max(K)*(1+(n_res-1)*u)/2
    min_m = max_m/2
    m = min_m + np.random.uniform(0,0.1,(n_com,n_spec))*(max_m - min_m)

    mu = np.sum(util*K,axis = -1) - m
    A = np.sum(util[:,np.newaxis]*util[:,:,np.newaxis]*K/r,
              axis = -1)
    return m,util,H, mu, A

i = 0
counter = 0
pars_LV = {}
pars_type1 = {}
while counter<n_com and timer()-start <= 1800:
    if i%n_com == 0:
        i = 0
        m,util,H, mu, A = new_community(n_com, n_spec)
    try:
        pars_LV = NFD_model(LV_model, n_spec, args = (mu[i],A[i]), xtol = 1e-5)
        NO_all[0,counter,:n_spec] = pars_LV["NO"].copy()
        FD_all[0,counter,:n_spec] = pars_LV["FD"].copy()
    except InputError:
        pass
    try:
        pars_type1 = NFD_model(mh.model,n_spec = n_spec, 
                         args = (mh.holling_type1, m[i], (util[i],)),
                                pars = pars_LV, xtol = 1e-5)
        NO_all[1,counter,:n_spec] = pars_type1["NO"].copy()
        FD_all[1,counter,:n_spec] = pars_type1["FD"].copy()
    except InputError:
        pass
    try:
        pars_type2 = NFD_model(mh.model,n_spec = n_spec, 
                         args = (mh.holling_type2, m[i], (util[i],H[i])), 
                                pars = pars_type1, xtol = 1e-5)
        
        NO_all[2,counter,:n_spec] = pars_type2["NO"]
        FD_all[2,counter,:n_spec] = pars_type2["FD"]
    except InputError:
        pass
    if np.any(np.isfinite(NO_all[:,counter])):
        counter += 1
    i += 1
    
    
print(np.sum(np.isfinite(NO_all), axis = -2))
np.savez("NFD_values,richness {}".format(n_spec), NO = NO_all, FD = FD_all)