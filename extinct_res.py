"""
compute the difference in NFD values for the mac Arthur resource model and
compare it to the NFD values of the corresponding LV -model

Resources may go extinct in the macArthur model, not however in the LV 
model, this is where the difference comes from"""

import numpy as np
import model_holling_discrete_res as mh
from nfd_definitions.numerical_NFD import NFD_model, InputError
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from timeit import default_timer as timer
start = timer()
n_spec = 3
n_com = 100

# resources
n_res = 5 #max(n_spec) # number of resources
id_res = np.arange(n_res) # identity of resources

K = np.ones(n_res) # carrying capacity of the resources
r = 2*np.ones(n_res) # regeneration speed of the resources

# to store NO and FD, for the 3 different models
NO_all, FD_all = np.full((2,2,n_com,n_spec), np.nan)
c = np.full((2,n_com), np.nan)

def LV_model(N,mu,A):
    return mu-A.dot(N)

def new_community(n_com, n_spec):
    # create new traits for a community
    u = 1/(2*n_res)
    util = u*2**np.random.uniform(-1,1,size = (n_com, n_spec, n_res))
    util[:,np.arange(n_spec), np.arange(n_spec)] = 1
    util[:,:,-1] = 0.1
    
    max_m = max(K)*(1+(n_res-1)*u)/2
    min_m = max_m/2
    m = min_m + np.random.uniform(0,0.1,(n_com,n_spec))*(max_m - min_m)

    mu = np.sum(util*K,axis = -1) - m
    A = np.sum(util[:,np.newaxis]*util[:,:,np.newaxis]*K/r,
              axis = -1)
    return m, util, mu, A

def res_model(N,util,m, ret = False):
    
    # compute density of resources
    R_star = K*(1-np.sum(util*N[:,np.newaxis],axis = 0)/r)
    R_star[R_star<0] = 0
    if not ret:      
        return np.sum(util*R_star,axis = -1) -m
    else:
        return R_star


m, util, mu, A = new_community(n_com,n_spec)

for i in range(n_com):
    try:
        pars_LV = NFD_model(LV_model, n_spec, args = (mu[i], A[i]))
    
        pars_res = NFD_model(res_model, n_spec, args = (util[i], m[i]))
    except InputError:
        continue
        print(i)
        fig, ax = plt.subplots(2,2,figsize = (7,7))
        time = np.linspace(0,50)
        y0 = [0.01,0]
        def res_wraper(N,t):
            return N*res_model(N,util[i], m[i])
        sol_0 = odeint(res_wraper, y0, time)
        y0 = [0,0.01]
        sol_1 = odeint(res_wraper, y0, time)
        res_0 = np.array([res_model(N, util[i], m[i], True) for N in sol_0])
        res_1 = np.array([res_model(N, util[i], m[i], True) for N in sol_1])
        ax[0,0].plot(time, sol_0)
        ax[0,1].plot(time, res_0)
        ax[1,0].plot(time, sol_1)
        ax[1,1].plot(time, res_1)
        continue
    
    NO_all[:,i] = pars_LV["NO"], pars_res["NO"]
    FD_all[:,i] = pars_LV["FD"], pars_res["FD"]
    c[:,i] = pars_LV["c"][0,1], pars_res["c"][0,1]
    
plt.figure()  
plt.scatter(NO_all[0,:,0], NO_all[1,:,0])
x_NO = np.linspace(np.nanmin(NO_all), np.nanmax(NO_all))
plt.plot(x_NO, x_NO)
"""
plt.figure()
plt.scatter(c[0], c[1])

plt.figure()  
plt.scatter(FD_all[0,:,0], FD_all[1,:,0])
identity = np.linspace(np.nanmin(FD_all), np.nanmax(FD_all))
plt.plot(identity, identity)

plt.figure()
plt.scatter(1-NO_all[0],-FD_all[0])
plt.plot(1-x_NO, (1-x_NO)/x_NO)

plt.figure()
plt.scatter(1-NO_all[1],-FD_all[1])
plt.plot(1-x_NO, (1-x_NO)/x_NO)"""