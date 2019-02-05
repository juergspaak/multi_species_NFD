import numpy as np
import matplotlib.pyplot as plt

from nfd_definitions.numerical_NFD import NFD_model

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
    return (real, A_prime[real], equi_prime[real],
                    sub_equi[sub_feasible & sub_stable])
    
def NFD_LV_multispecies(A,sub_equi, r = 1):
    # compute the two species niche overlap
    # sqrt (a_ij*a_ji/(a_ii*a_jj))
    NO_ij = np.sign(A)*np.sqrt(np.abs(
                np.einsum("nij,nji,nii,njj->nij",A,A,1/A,1/A)))
    
    # NO is a weighted average of the two species NO
    # weights = sqrt(a_ij/a_ji*a_jj)
    weights = np.sqrt(np.abs(np.einsum("nij,nji,njj->nij",A,1/A,A)))*sub_equi
    NO = np.average(NO_ij, axis = -1, weights = weights)
    
    # (1-F_i^j)*a_jj = sqrt(a_ij/a_ji*a_ii*a_jj)
    summands = np.sqrt(np.abs(np.einsum("nij,njj,nji,nii->nij",A,A,1/A,A)))
    FD = 1- 1/r*np.sum(summands*sub_equi, axis = -1)
    
    return NO, FD

def diag_fill(A, values):
    n = A.shape[-1]
    A[:, np.diag_indices(n)[0], np.diag_indices(n)[1]] = values
    return

NO_all = []
FD_all = []
A_all = []
equi_all = []
equi_mono = []
n_com_prime = 1000 # number of communities at the beginning
max_alpha = 0.5
min_alpha = 0
mu = 1

diag_one = True
symm = False
title = "{}<a<{}".format(min_alpha, max_alpha)
if diag_one:
        title += "; a_ii = 1"
if symm:
        title += "; symm"
# number of species ranging from 2 to 7
for n in range(2,11):
    # create random interaction matrices
    A_prime = np.random.uniform(min_alpha,max_alpha,size = (n_com_prime,n,n))
    
    # intraspecific competition is assumed to be 1
    diag_fill(A_prime, np.random.uniform(1,2,(n_com_prime,n)))
    if diag_one:
        diag_fill(A_prime,1)
    if symm:
        A_prime = (A_prime + A_prime.swapaxes(1,2))/2
    # intrinsic growth rate
    r_prime = np.ones((n_com_prime,n))
    
    real, A, equi, sub_equi = find_real_communities(A_prime, r_prime)
    NO, FD = NFD_LV_multispecies(A,sub_equi)
    print(len(NO),n)
    NO_all.append(NO)
    FD_all.append(FD)
    A_all.append(A)
    equi_mono.append(mu/A[:,np.diag_indices(n)[0], np.diag_indices(n)[0]])
    equi_all.append(equi)

# compute relative yield
ry = [equi_all[i]/equi_mono[i] for i in range(len(equi_all))]

# check result with random index
n_test = np.random.randint(len(A_all))
test_ind = np.random.randint(len(A_all[n_test]))
def test_f(N):
    return 1 - np.dot(A_all[n_test][test_ind],N)
    
pars = NFD_model(test_f, n_test+2)
print(pars["NO"])
print(NO_all[n_test][test_ind])
print(FD_all[n_test][test_ind])
print(pars["FD"])

def bounds(array, percents =  [5,95]):
    start = np.array([np.percentile(arr,percents) for arr in array])
    return np.array([min(start[:,0]), max(start[:,1])])
FD_bounds = bounds(FD_all)
NO_bounds = bounds(NO_all)
ND_bounds = 1-NO_bounds[::-1]
ry_bounds = bounds(ry)
ryt = [np.sum(ry_i, axis = -1) for ry_i in ry]
ryt_bounds = bounds(ryt)

###############################################################################
# plot the results

fs = 14

# NO and FD versus species richness    
fig = plt.figure(figsize = (11,11))
ax_NO = fig.add_subplot(2,2,1)
ax_NO.boxplot(NO_all, positions = range(2,11), showfliers = False)
ax_NO.set_ylabel(r"$\mathcal{NO}$")

ax_FD = fig.add_subplot(2,2,3, sharex = ax_NO)
ax_FD.boxplot(FD_all, positions = range(2,11), showfliers = False)
ax_FD.set_xlabel("number of species")
ax_FD.set_ylabel(r"$\mathcal{F}$", fontsize = fs)

# effect of ND and FD on relative yield

ax_coex = fig.add_subplot(1,2,2)
x = np.linspace(0,1,1000)
for i in range(len(NO_all)):
    im = ax_coex.scatter(1-NO_all[i][:,0], FD_all[i][:,0], s = 5, linewidth = 0,
                c = ry[i][:,0], vmin = ry_bounds[0], vmax = ry_bounds[1])
ax_coex.plot(x,-x/(1-x), color = "black")
ax_coex.set_xlim(ND_bounds)
ax_coex.set_ylim(FD_bounds)
plt.gca().invert_yaxis()
ax_coex.set_ylabel(r"$-\mathcal{F}$", fontsize = fs)
ax_coex.set_xlabel(r"$\mathcal{N}$", fontsize = fs)
cbar = fig.colorbar(im,ax = ax_coex)
cbar.ax.set_ylabel("relative yield")

fig.savefig("Figure, NFD effect on RYT,{},{},{},{}.pdf".format(
        min_alpha, max_alpha, symm, diag_one))
#"""
