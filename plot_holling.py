import numpy as np
from scipy.integrate import simps

import model_holling as mh
from nfd_definitions.numerical_NFD import NFD_model, InputError

import sys
from timeit import default_timer as timer
start = timer()
n_spec = int(sys.argv[1])
n_com = 1000


# to store NO and FD, for the 3 different models
NO_all, FD_all = np.full((2,3,n_com,n_spec), np.nan)

def LV_model(N,mu,A):
    return mu-A.dot(N)

def new_community(n_com, n_spec):
    
    traits = np.random.uniform(size = (n_com, n_spec, 1))
    max_util = np.random.uniform(1,2,size = (n_com, n_spec, 1))
    
    util_width = np.random.uniform(0.1,0.2,size = (n_com,n_spec,1))
    rel_util = np.exp(-(traits-mh.id_res)**2/(2*util_width**2))
    H = 1 + np.exp(-(traits-mh.id_res)**2/(2*util_width**2))
    
    # utilisation function
    util = rel_util * max_util
    
    min_m = max_util[...,0]*(np.sqrt(2)-1)*np.sqrt(np.pi)*util_width[...,0]
    max_m = max_util[...,0]*max(mh.K)*np.sqrt(2*np.pi)*util_width[...,0]
    m = min_m + np.random.uniform(0,0.1,n_spec)*(max_m - min_m)

    mu = simps(util*mh.K,axis = -1, dx = mh.d_id) - m
    A = simps(util[:,np.newaxis]*util[:,:,np.newaxis]*mh.K/mh.r,
              axis = -1, dx = mh.d_id)
    return m,util,H, mu, A

i = 0
counter = 0 
while counter<n_com or timer()-start >= 1800:
    if i%n_com == 0:
        i = 0
        m,util,H, mu, A = new_community(n_com, n_spec)
    try:
        pars_LV = NFD_model(LV_model, n_spec, args = (mu[i],A[i]))
        NO_all[0,counter,:n_spec] = pars_LV["NO"].copy()
        FD_all[0,counter,:n_spec] = pars_LV["FD"].copy()
    except InputError:
        pass
    try:
        pars_type1 = NFD_model(mh.model,n_spec = n_spec, 
                         args = (mh.holling_type1, m[i], (util[i],)),
                                pars = pars_LV)
        NO_all[1,counter,:n_spec] = pars_type1["NO"].copy()
        FD_all[1,counter,:n_spec] = pars_type1["FD"].copy()
    except InputError:
        pass
    try:
        pars_type2 = NFD_model(mh.model,n_spec = n_spec, 
                         args = (mh.holling_type2, m[i], (util[i],H[i])), 
                                pars = pars_type1)
        
        NO_all[2,counter,:n_spec] = pars_type2["NO"]
        FD_all[2,counter,:n_spec] = pars_type2["FD"]
    except InputError:
        pass
    if np.any(np.isfinite(NO_all[:,counter])):
        counter += 1
    i += 1
    
    
    
np.savez("NFD_values,richness {}".format(n_spec), NO = NO_all, FD = FD_all)