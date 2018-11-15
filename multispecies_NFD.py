import numpy as np
import matplotlib.pyplot as plt
from numerical_NFD import find_NFD

def find_real_communities(A_prime,r_prime):
    
    # compute equilibrium densities of entire community
    equi_prime = np.linalg.solve(A_prime,r_prime)
    # corresponds to diagonal matrix of species densities
    diag_equi_prime = equi_prime.reshape(len(A_prime),1,n)
    
    # find feasible (N^*>0) and stable (negative eigenvalues of Jacobi) com.
    feasible = np.all(equi_prime>0,axis = 1)
    stable = np.all(np.real(np.linalg.eigvals(-diag_equi_prime*A_prime))<0,
                    axis = -1)
    
    # retain only feasible and stable communities
    A = A_prime[feasible & stable]
    equi = equi_prime[feasible & stable]
    n_com = len(A)
    
    # compute equilibrium densities where one species is absent
    sub_equi = np.zeros((n_com,n,n))
    sub_jacobi = np.empty((n_com,n,n-1), dtype = "complex")
    
    for i in range(n):
        # to remove species i
        inds = np.arange(n)[np.arange(n)!=i]
        # compute subcommunities equilibrium
        sub_equi[:,i,inds] = np.linalg.solve(A[:,inds[:,np.newaxis],inds],
                        np.ones((n_com,n-1)))
        # compute stability of subcommunities
        diag_sub_equi = sub_equi[:,i,inds].reshape(n_com,1,n-1)
        sub_jacobi[:,i] = np.linalg.eigvals(-diag_sub_equi
                            *A[:,inds[:,np.newaxis],inds])
    
    # all subcommunities must be stable and feasible
    # sub_equi[:,i,i] = 0, all others must be positive
    sub_feasible = np.sum(sub_equi>0, axis = (1,2)) == n*(n-1)
    sub_stable = np.all(np.real(sub_jacobi)<0, axis = (1,2))
    
    real = feasible & stable
    real[real] = sub_feasible & sub_stable
    return real, A_prime[real], equi, sub_equi[sub_feasible & sub_stable]
    
def NFD_LV_multispecies(A,sub_equi, r = 1):
    # compute the two species niche overlap
    NO_ij = np.sqrt(A*A.swapaxes(1,2))
    
    # NO is a weighted average of the two species NO
    NO = np.average(NO_ij, axis = -1, 
               weights = np.sqrt(A/A.swapaxes(1,2))*sub_equi)
    
    FD = 1- 1/r*np.sum(np.sqrt(A/A.swapaxes(1,2))*sub_equi, axis = -1)
    
    return NO, FD

NO_all = []
FD_all = []
n_com_prime = 10000 # number of communities at the beginning

# number of species ranging from 2 to 7
for n in range(2,11):
    # create random interaction matrices
    A_prime = np.random.uniform(0,0.2,size = (n_com_prime,n,n))
    # intraspecific competition is assumed to be 1
    A_prime[:, np.diag_indices(n)[0], np.diag_indices(n)[1]] = 1
    # intrinsic growth rate
    r_prime = np.ones((n_com_prime,n))
    
    real, A, equi, sub_equi = find_real_communities(A_prime, r_prime)
    NO, FD = NFD_LV_multispecies(A,sub_equi)
    print(len(NO),n)
    NO_all.append(NO)
    FD_all.append(FD)
    
plt.figure()
plt.boxplot(NO_all, positions = range(2,11), showfliers = False)
plt.xlabel("number of species")
plt.ylabel("NO")

plt.figure()
plt.boxplot(FD_all, positions = range(2,11), showfliers = False)
plt.xlabel("number of species")
plt.ylabel("FD")
plt.ylim([-10,1])
"""
# check result with random index
i = np.random.randint(len(A))
def test_f(N):
    return 1 - np.dot(A[i],N)
    
pars = find_NFD(test_f, n)
print(pars["NO"])
print(NO[i])
print(FD[i])
print(pars["FD"])"""
