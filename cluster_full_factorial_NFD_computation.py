import numpy as np

import sys
from higher_order_models import NFD_higher_order_LV
from interaction_estimation import resample
from timeit import default_timer as timer

# determine string and parameter settings for run   
interaction = ["neg, ", "bot, ", "pos, "] # 1. order interaction
ord_2 = ["neg, ", "bot, ", "pos, ", "abs, "] # 2. order interaction
ord_3 = ["pre, ", "abs, "] # presence of third order interaction
correlation = ["pos, ", "neg, ", "nul, "]
connectance = ["h, ", "m, ", "l, "] # connectance

strings = [i+j+k+l+m for i in interaction for j in ord_2 for k in ord_3
           for l in connectance for m in correlation]
interaction = [[-1,0], [-1,1], [0,1]]
ord_2 = [[-1,0], [-1,1], [0,1], [0,0]]
ord_3 = [[-1,1],[0,0]]
correlation = [1,-1,0]
connectance = [1, 4/5, 2/3]

parameters = [[i,j,k,l,m] for i in interaction for j in ord_2 for k in ord_3
           for l in connectance for m in correlation]
# try getting parameters from jobscript
try:
    job_id = int(sys.argv[1])-1
    n_com = 100 # number of communities at the beginning
except IndexError:
    job_id = np.random.randint(len(strings))
    n_com = 100 # number of communities at the beginning

string = strings[job_id]
parameters = parameters[job_id]
parameters = [[-1,0], [-1,0], [0,0], 1, 0]
keys = ["ord1", "ord2", "ord3", "con", "cor"]
parameters = {key: parameters[i] for i, key in enumerate(keys)}
print(parameters)
beta = gamma = 0.05

n_order = 3

richness = np.arange(2,7)
mu = np.ones((n_com, richness[-1]))

NO_all, FD_all, NO_all_no_indir, FD_all_no_indir = np.full((4,len(richness),
                                                    n_com,richness[-1]),np.nan)
c_all, c_all_no_indir = np.full((2,len(richness), n_com, richness[-1],
                                 richness[-1]), np.nan)
interaction = [0,0]
conns = [0,0]
fac = 10 # probability of not being connected is 2.6% in the worst case
start = timer()
A_all, B_all, C_all = np.full((3, len(richness),n_com) + 4*(richness[-1],),
                           np.nan)
A_all = A_all[...,0,0].copy() # reduce dimension of A
B_all = B_all[...,0].copy() # reduce dimension
for r,n in enumerate(richness):
    print(r,n)
    ind_u = np.triu_indices(n, 1) # indices of a_ij, i<j
    ind_l = np.tril_indices(n) # indices od a_ij, i>j
    
    # baseline parameters, effects on the interactions
    # Create interaction matrix first
    # resample interaction distribution
    aij = resample(5*fac*n_com*n*n) # create to many
    aij = aij[(aij>parameters["ord1"][0]) # minimal value
                & (aij<parameters["ord1"][1])] # maximal value
    A = aij[:fac*n_com*n*n].reshape(fac*n_com, n,n) # reshape into matrix
    
    # change correlation between species
    if parameters["cor"] == 1: # same entries
        A[:, ind_u[1], ind_u[0]] = A[:, ind_u[0], ind_u[1]]
    elif parameters["cor"] == -1:
        A[:, ind_u[1], ind_u[0]] = - A[:, ind_u[0], ind_u[1]]
        A[:, ind_u[1], ind_u[0]] += np.mean(aij)
    # change connectance between species
    conn = np.random.uniform(size = A.shape)
    conn[:, ind_l[0], ind_l[1]] = np.nan

    linked = conn <= np.nanpercentile(conn, 100 * parameters["con"],
                                      axis =(1,2), keepdims = True)
    conn[linked] = 1
    conn[~linked] = 0
    conn[:, ind_u[1], ind_u[0]] = conn[:, ind_u[0], ind_u[1]]
    conn[:, np.arange(n), np.arange(n)] = 1 # sp interact with themselves
    
    A *= conn
    
    # remove interaction matrices that are not connected by computing laplacian
    laplacian = -1.0 * (A !=0)
    laplacian[:, np.arange(n), np.arange(n)] = np.sum(A !=0,
              axis = 1) - 1 # remove intraspecific link
    eig = np.linalg.eigvalsh(laplacian)
    connected = eig[:,1] > 1e-10 # connected if second smallest eigv > 0
    if sum(connected)< n_com: # did not find sufficiently many connected coms.
        np.savez("no_com{}.npz".format(string), A = A)
        raise ValueError("not enough good graphs")
        
    A = A[connected][:n_com]
    A[:,np.arange(n), np.arange(n)] = 1 # set intraspecific effects
    con_lower = conn[connected][:n_com]
    
    # create higher order interactions
    for i, key in enumerate(["ord2", "ord3"]):
        inter = np.random.uniform(*parameters[key],
                size = (n_com, n) + int(key[-1])*(n,))
        # change correlation between species
        if parameters["cor"] == 1: # same entries
            inter[:, ind_u[1], ind_u[0]] = inter[:, ind_u[0], ind_u[1]]
        elif parameters["cor"] == -1:
            inter[:, ind_u[1], ind_u[0]] = - inter[:, ind_u[0], ind_u[1]]
            inter[:, ind_u[1], ind_u[0]] += sum(parameters[keys[i]])
        
        
        # change connectance between species
        conn = np.random.uniform(size = inter.shape)
        conn[:, ind_l[0], ind_l[1]] = np.nan
        # can't affect not interacting species
        conn[con_lower[...,np.newaxis] == np.zeros(n)] = np.nan
        linked = conn <= np.nanpercentile(conn, 100 * parameters["con"],
                                          axis =(1,2), keepdims = True)
        conn[linked] = 1
        conn[~linked] = 0
        conn[:, ind_u[1], ind_u[0]] = conn[:, ind_u[0], ind_u[1]]
        conn[:, np.arange(n), np.arange(n)] = 1 # sp interact with themselves
        conns[i] = conn.copy()
        inter *= conn
        con_lower = conn
        interaction[i] = inter
        
    B = A[...,np.newaxis] * interaction[0] * beta # expand multiplication
    C = B[...,np.newaxis] * interaction[1] * gamma# expand multiplication
    
    # ensure that specie have negative effect on themselves
    ns = np.arange(n)
    B[:, ns, ns, ns] = -np.abs(B[:, ns, ns, ns])
    C[:, ns, ns, ns, ns] = -np.abs(C[:, ns, ns, ns, ns])
    
    interactions = [A,B,C]
    A_all[r, :, :n, :n] = A.copy()
    B_all[r, :, :n, :n, :n] = B.copy()
    C_all[r, :, :n, :n, :n, :n] = C.copy()
    NO,FD,c, NO_no_indir,FD_no_indir,c_no_indir =\
                    NFD_higher_order_LV(mu[:,:n],*interactions)
    NO_all[r,:len(NO),:n] = NO
    FD_all[r,:len(FD),:n] = FD
    c_all[r,:len(c),:n,:n] = c
    NO_all_no_indir[r,:len(NO),:n] = NO_no_indir
    FD_all_no_indir[r,:len(FD),:n] = FD_no_indir
    c_all_no_indir[r,:len(c),:n,:n] = c_no_indir
    print(timer()-start)

np.savez("NFD_val/NFD_values {}".format(string),
         FD = FD_all, ND = 1-NO_all, c = c_all, parameters = parameters,
         FD_no_indir = FD_all_no_indir, ND_no_indir = 1- NO_all_no_indir,
         c_no_indir = c_all_no_indir,
         A = A_all, B = B_all, C = C_all)
print(np.isfinite(NO_all[...,0]).sum(axis = 1))


