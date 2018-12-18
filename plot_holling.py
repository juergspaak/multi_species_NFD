import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

import model_holling as mh
from nfd_definitions.numerical_NFD import NFD_model, InputError


richness = np.arange(2,5, dtype = "int")
n_com = 100


# to store NO and FD, for the 3 different models
NO_all, FD_all = np.full((2,3,len(richness),n_com,max(richness)), np.nan)

def LV_model(N,mu,A):
    return mu-A.dot(N)

for r, n_spec in enumerate(richness):
    print(r)
    n_spec = int(n_spec)
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
    for i in range(n_com):
        try:
            pars_LV = NFD_model(LV_model, n_spec, args = (mu[i],A[i]))
            NO_all[0,r,i,:n_spec] = pars_LV["NO"].copy()
            FD_all[0,r,i,:n_spec] = pars_LV["FD"].copy()
        except InputError:
            pass
        try:
            pars_type1 = NFD_model(mh.model,n_spec = n_spec, 
                             args = (mh.holling_type1, m[i], (util[i],)),
                                    pars = pars_LV)
            NO_all[1,r,i,:n_spec] = pars_type1["NO"].copy()
            FD_all[1,r,i,:n_spec] = pars_type1["FD"].copy()
        except InputError:
            pass
        try:
            pars_type2 = NFD_model(mh.model,n_spec = n_spec, 
                             args = (mh.holling_type2, m[i], (util[i],H[i])), 
                                    pars = pars_type1)
            
            NO_all[2,r,i,:n_spec] = pars_type2["NO"]
            FD_all[2,r,i,:n_spec] = pars_type2["FD"]
        except InputError:
            pass
        
fs = 14    
fig = plt.figure(figsize = (11,11))
n_models = 3
for m in range(n_models):
    
    if m == 0:
        ax_NO = fig.add_subplot(6,n_models,m+1)
        ax_FD = fig.add_subplot(6,n_models,m+n_models+1)
    else:
        ax_NO = fig.add_subplot(6,n_models,m+1, sharey = ax_NO)
        ax_FD = fig.add_subplot(6,n_models,m+n_models+1, sharey = ax_FD)
    ax_NO.boxplot([NO[np.isfinite(NO)] for NO in NO_all[m,...,0]],
                  positions = richness,showfliers = False)

    
    ax_FD.boxplot([FD[np.isfinite(FD)] for FD in FD_all[m,...,0]],
                  positions = richness, showfliers = False)
    ax_FD.set_xlabel("number of species")