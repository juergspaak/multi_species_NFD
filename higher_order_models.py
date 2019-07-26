
import numpy as np
from nfd_definitions.numerical_NFD import NFD_model, InputError

n = 2
n_com = 100

mu = 1
alpha,beta,gamma = 0.1,0.1,0.1
A = alpha * np.random.normal(size = (n_com, n,n))
B = beta * np.random.normal(size = (n_com, n,n,n))
C = gamma * np.random.normal(size = (n_com, n,n,n,n))

alpha,beta,gamma = 0.01,0.01,0.01
A = -alpha * np.random.uniform(-1,1,size = (n_com, n,n))
B = -beta * np.random.uniform(-1,1,size = (n_com, n,n,n))
C = -gamma * np.random.uniform(-1,1,size = (n_com, n,n,n,n))
A[:,np.arange(n), np.arange(n)] = -1
B[:,np.arange(n), np.arange(n), np.arange(n)] = 0
C[:,np.arange(n), np.arange(n), np.arange(n), np.arange(n)] = 0

A_def = np.zeros((n,n))
B_def = np.zeros((n,n,n))
C_def = np.zeros((n,n,n,n))

def interaction(T,N,d):
    return np.tensordot(T,tensor(N,d-1),axes = d-1)

def tensor(N,d):
    return np.prod([N.reshape((-1,)+i*(1,))*np.ones(d*(len(N),)) 
                    for i in range(d)], axis = 0)

def LV_model(N, A = A_def, B = B_def, C = C_def):
    return mu - interaction(A,N,2) - interaction(B,N,3) - interaction(C,N,4)


index = np.full(n_com,False, dtype = "bool")

def NFD_higher_order_LV(mu,A,B = None, C = None):
    n_com, n = A.shape[:2]
    if B is None:
        B = np.zeros(A.shape + (n,))
    if C is None:
        C = np.zeros(A.shape + (n,n))
    
    NO,FD = np.empty((2,n_com,n))
    c = np.empty((n_com, n,n))
    index = np.full(n_com,False,dtype = "bool")
    
    # compute sub-community equilibrium based on 1. order interaction
    sub_equi = np.zeros((n_com,n,n))
    
    for i in range(n):
        # to remove species i
        inds = np.arange(n)[np.arange(n)!=i]
        # compute subcommunities equilibrium
        sub_equi[:,i,inds] = np.linalg.solve(
                A[:,inds[:,np.newaxis],inds],mu[:, inds])
    c = np.sqrt(np.abs(np.einsum("nij,nji-> nij", A, 1/A)))
    c[np.isnan(c)] = 1
    c[~np.isfinite(c)] = 1
    c[c == 0] = 1
    pars = {}
    for i in range(n_com):
        try:
            # assume c and equilibrium densities based on first order
            pars["c"] = c[i]
            pars["N_star"] = sub_equi[i]
            pars = NFD_model(LV_model,n,args = (A[i],B[i], C[i]), pars = pars)
            index[i] = True
        except InputError:
            continue
        NO[i] = pars["NO"]
        FD[i] = pars["FD"]
        c[i] = pars["c"]
    return NO[index], FD[index], c[index]