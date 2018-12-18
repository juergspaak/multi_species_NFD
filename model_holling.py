"""
@author: J.W.Spaak
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import simps,odeint

from timeit import default_timer as timer


n_res = 101 # number of resources
id_res, d_id = np.linspace(-1,2,n_res, retstep = True) # identity of resources

K = np.ones(n_res) # carrying capacity of the resources
r = np.ones(n_res) # regeneration speed of the resources

def solver_jurg(f,a,b, args = (), rel_tol = 1e-8, abs_tol = 1e-12):
    a,b = a.copy(),b.copy()
    if not ((f(a,*args)<0).all() and (f(b,*args)>0).all()):
        #raise ValueError("JSP,f(a) must be negative and f(b) must be positive")
        pass
    rel_diff, abs_diff = np.inf, np.inf
    counter = 0
    while rel_diff > rel_tol and abs_diff > abs_tol :
        change_b = f((a+b)/2,*args)>0
        b[change_b] = ((a+b)/2)[change_b]
        a[~change_b] = ((a+b)/2)[~change_b]
        rel_diff = np.amax(np.abs(b-a)/a)
        abs_diff = np.amax(np.abs(b-a))
        counter +=1
        """if not ((f(a,*args)<0).all() and (f(b,*args)>0).all()):
            print(counter,min(f(b,*args)),max(f(a,*args)))
            raise ValueError("JSP,f(a) must be negative and f(b) "
                             "must be positive")"""
    return (a+b)/2

def res_dens(N,holling,spec_args = (),R_args = (r,K,d_id)):
    r, K, d_id = R_args
    
    
    # resources that have been depleted
    dead_res = (r-np.sum(holling(N,0,*spec_args)*N[:,np.newaxis],axis = 0))<=0
    if dead_res.any():
        # raise ValueError("heeelp")
        pass
        
    def dR_dt(R):
        return r*(1-R/K) - np.sum(holling(N, R, *spec_args)*N[:,np.newaxis]
                  ,axis = 0)                   
    R_star = solver_jurg(dR_dt, K*(1+1e-8), np.zeros(K.shape))
    return R_star

def holling_type1(N, R, util):
    return util

def holling_type2(N, R, util, H):
    return util/(R+H)

def model(N, holling, m, spec_args = (), R_args = (r,K,d_id)):
    R_star = res_dens(N, holling, spec_args, R_args)
    return simps(R_star*holling(N,R_star,*spec_args), dx = d_id, axis = -1)-m

def r_star_check(K,r,util,N):
    return K*(1-np.sum(util*N[:,None],axis = 0)/r)

n_spec = 5 # number of species
traits = np.random.uniform(size = (n_spec,1)) # trait to identify the species
max_util = np.random.uniform(1,2,size = (n_spec,1))
util_width = np.random.uniform(0.1,0.2,size = (n_spec,1))
rel_util = np.exp(-(traits-id_res)**2/(2*util_width**2))
H = 1 + np.exp(-(traits-id_res)**2/(2*util_width**2))

# utilisation function
util = rel_util * max_util

min_m = max_util[:,0]*(np.sqrt(2)-1)*np.sqrt(np.pi)*util_width[:,0]
max_m = max(K)*np.sqrt(2*np.pi)*util_width[:,0]
m = min_m + np.random.uniform(0,1,n_spec)*(max_m - min_m)

A = simps(util*util[:,np.newaxis]*K/r, axis = -1, dx = d_id)
mu = simps(util*K,axis = -1, dx = d_id) - m
"""        
for i in range(10):
    N = np.random.uniform(0,0.1,n_spec)
    LV = mu - A.dot(N)
    hol = model(N, holling_type1, m, spec_args =  (util,))
    hol_type2 = model(N,holling_type2, m, (util, H))
    print(hol/LV, hol_type2/LV)
    
plt.figure()
plt.plot(util.T)

N_start = mu/np.diag(A)/10

time = np.linspace(0,50,50)
start = timer()
sol_LV = odeint(lambda N,t: N*(mu-A.dot(N)),N_start, time)
end = timer()
sol_holling = odeint(lambda N,t: N*model(N, holling_type1, m, (util,)), 
                     N_start, time)
end2 = timer()
plt.figure()
plt.plot(time, sol_LV)
plt.figure()
plt.plot(time, sol_holling)

time = np.linspace(0,50,50)
sol_holling2 = odeint(lambda N,t: N*model(N, holling_type2, m, (util,H)), 
                     N_start, time)

plt.figure()
plt.plot(time, sol_holling2)"""