"""
@author: J.W.Spaak
"""

import numpy as np

n_res = 10 # number of resources
id_res = np.arange(n_res) # identity of resources

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

def res_dens(N,holling,spec_args = (),R_args = (r,K)):
    r, K = R_args
    
    def dR_dt(R):
        return r*(1-R/K) - np.sum(holling(N, R, *spec_args)*N[:,np.newaxis]
                  ,axis = 0)                   
    R_star = solver_jurg(dR_dt, K*(1+1e-8), np.zeros(K.shape))
    return R_star

def holling_type1(N, R, util):
    return util

def holling_type2(N, R, util, H):
    return util/(R+H)

def model(N, holling, m, spec_args = (), R_args = (r,K)):
    R_star = res_dens(N, holling, spec_args, R_args)
    return np.sum(R_star*holling(N,R_star,*spec_args), axis = -1)-m

def r_star_check(K,r,util,N):
    return K*(1-np.sum(util*N[:,None],axis = 0)/r)