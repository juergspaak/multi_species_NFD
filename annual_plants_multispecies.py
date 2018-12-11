import numpy as np
import matplotlib.pyplot as plt

from NFD_code.numerical_NFD import NFD_model, InputError

def find_real_communities(A_prime,lamb_prime):
    n = A_prime.shape[-1]
    # compute equilibrium densities of entire community
    equi_prime = np.linalg.solve(A_prime,lamb_prime)
    
    # find feasible (N^*>0) and stable (negative eigenvalues of Jacobi) com.
    feasible = np.all(equi_prime>0,axis = 1)
    
    # retain only feasible and stable communities
    A = A_prime[feasible]
    lamb = lamb_prime[feasible]
    n_com = len(A)
    
    # compute equilibrium densities where one species is absent
    sub_equi = np.zeros((n_com,n,n))
    
    for i in range(n):
        # to remove species i
        inds = np.arange(n)[np.arange(n)!=i]
        # compute subcommunities equilibrium
        sub_equi[:,i,inds] = np.linalg.solve(A[:,inds[:,np.newaxis],inds],
                        lamb[:,inds])
    
    # all subcommunities must be stable and feasible
    # sub_equi[:,i,i] = 0, all others must be positive
    sub_feasible = np.sum(sub_equi>0, axis = (1,2)) == n*(n-1)
    
    feasible[feasible] = sub_feasible
    return feasible, sub_equi[sub_feasible]
    
def diag_fill(A, values):
    n = A.shape[-1]
    A[:, np.diag_indices(n)[0], np.diag_indices(n)[1]] = values
    return

def NFD_annual_plants(A,lamb,model):
    richness = A.shape[-1]
    # find which communities actually have a stable sub_community
    feasible, sub_equi = find_real_communities(A,model.lamb_eq(lamb))
    sub_equi = model.N_eq(sub_equi)
    
    A = A[feasible]
    lamb = lamb[feasible]
    
    NO,FD = np.empty((2,len(A),richness))
    index = np.full(len(A),True, dtype = "bool")
    for j in range(len(A)):
        try:
            pars = NFD_model(model.model,richness, args = (A[j],lamb[j]),
                             pars = {"N_star": sub_equi[j]})
            NO[j] = pars["NO"]
            FD[j] = pars["FD"]
        except InputError:
            index[j] = False
            print(j, "j")
            print(sub_equi[j],"sub_equi\n\n",A[j], "A\n\n", lamb[j], "lambda")
            raise
    print(sum(index))
    return NO[index], FD[index]