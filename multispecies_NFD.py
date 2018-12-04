import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

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
    NO_ij = np.sqrt(np.abs(A*A.swapaxes(1,2)))*np.sign(A)
    
    # NO is a weighted average of the two species NO
    NO = np.average(NO_ij,axis = -1,
                    weights = np.sqrt(np.abs(A/A.swapaxes(1,2)))*sub_equi)
    
    FD = 1- 1/r*np.sum(np.sqrt(np.abs(A/A.swapaxes(1,2)))*sub_equi, axis = -1)
    
    return NO, FD

NO_all = []
FD_all = []
equi_all = []
n_com_prime = 1000 # number of communities at the beginning
max_alpha = 0.2
min_alpha = 0

# number of species ranging from 2 to 7
for n in range(2,11):
    # create random interaction matrices
    A_prime = np.random.uniform(-min_alpha,max_alpha,size = (n_com_prime,n,n))
    # intraspecific competition is assumed to be 1
    A_prime[:, np.diag_indices(n)[0], np.diag_indices(n)[1]] = 1
    # intrinsic growth rate
    r_prime = np.ones((n_com_prime,n))
    
    real, A, equi, sub_equi = find_real_communities(A_prime, r_prime)
    NO, FD = NFD_LV_multispecies(A,sub_equi)
    print(len(NO),n)
    NO_all.append(NO)
    FD_all.append(FD)
    equi_all.append(equi)


from numerical_NFD import NFD_model
# check result with random index
i = np.random.randint(len(A))
def test_f(N):
    return 1 - np.dot(A[i],N)
    
pars = NFD_model(test_f, n)
print(pars["NO"])
print(NO[i])
print(FD[i])
print(pars["FD"])

###############################################################################
# plot the results

# NO and FD versus species richness    
fig_box, ax = plt.subplots(2,1,sharex = True)
ax[0].boxplot(NO_all, positions = range(2,11), showfliers = False)
ax[0].set_ylabel(r"$\mathcal{NO}$")

ax[1].boxplot(FD_all, positions = range(2,11), showfliers = False)
ax[1].set_xlabel("number of species")
ax[1].set_ylabel(r"$\mathcal{F}$", fontsize = 16)
ax[1].set_ylim([-10,1])
fig_box.savefig("Figure, species richness on NFD.pdf")

# FD and NO and species equilibrium density
fig, ax = plt.subplots(3,3, sharex = True, sharey = True, figsize = (9,9))
x = np.linspace(0,1,1000)
for i in range(len(NO_all)):
    axc = ax.flatten()[i]
    im = axc.scatter(1-NO_all[i][:,0], -FD_all[i][:,0], s = 5, linewidth = 0,
                c = equi_all[i][:,0], vmin = 0.2, vmax = 2)
    axc.set_title(i+2)
    axc.plot(x,x/(1-x))
axc.set_xlim(1-max_alpha, 1+min_alpha)
axc.set_ylim(-1,10)
ax[1,0].set_ylabel(r"$-\mathcal{F}$", fontsize = 20)
ax[2,1].set_xlabel(r"$\mathcal{N}$", fontsize = 20)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.savefig("Figure, NFD effect on density.pdf")

# same as above, but all in one figure
fig = plt.figure()
x = np.linspace(0,1,1000)
for i in range(len(NO_all)):
    im = plt.scatter(1-NO_all[i][:,0], -FD_all[i][:,0], s = 5, linewidth = 0,
                c = equi_all[i][:,0], vmin = 0.2, vmax = 2)
plt.plot(x,x/(1-x), color = "black")
plt.xlim(1-max_alpha, 1+min_alpha)
plt.ylim(-1,10)
plt.ylabel(r"$-\mathcal{F}$", fontsize = 20)
plt.xlabel(r"$\mathcal{N}$", fontsize = 20)
plt.colorbar(im)
fig.savefig("Figure, NFD effect on density_one plot.pdf")

# prediction of equilibrium density via NFD
fig = plt.figure()
for i in range(len(NO_all)):
    im = plt.scatter((1+NO_all[i]*(FD_all[i]-1))/(1-NO_all[i]**2), equi_all[i], s = 5, 
                     linewidth = 0)
plt.xlabel(r"$\mathcal{N}+\mathcal{F}-\mathcal{NF}$", fontsize = 16)
plt.ylabel(r"$N^*_i$", fontsize = 16)
fig.savefig("Figure, prediciton of N_star via NFD.pdf")

# FD and equilibrium density in community
fig, ax = plt.subplots(3,3, sharex = True, sharey = True, figsize = (9,9))
x = np.linspace(0,1,1000)
for i in range(len(NO_all)):
    axc = ax.flatten()[i]
    axc.scatter(np.average(1-NO_all[i], axis = -1), np.sum(equi_all[i], 
                           axis = -1), s = 5, linewidth = 0)
    axc.set_title(i+2)
axc.set_xlim(1-max_alpha,1+min_alpha)
ax[2,1].set_xlabel(r'$\overline{\mathcal{N}}$', fontsize = 20)
ax[1,0].set_ylabel("EF", fontsize = 20)
fig.savefig("Figure, ND effect on EF.pdf")

# prediction of EF via ND and FD
# FD and equilibrium density in community
fig, ax = plt.subplots(3,3, sharex = True, sharey = True, figsize = (9,9))
x = np.linspace(0,1,1000)
for i in range(len(NO_all)):
    axc = ax.flatten()[i]
    axc.scatter(np.sum((1+NO_all[i]*(FD_all[i]-1)), axis = -1), 
                np.sum(equi_all[i], axis = -1), s = 5, linewidth = 0)
    axc.set_title(i+2)
ax[2,0].set_xlabel(r'$\overline{\mathcal{N}}+\overline{\mathcal{F}}-\overline{\mathcal{NF}}$', fontsize = 20)
ax[2,2].set_xlabel(r'$=n\left(1-NO(1-F)+cov(NO,F)\right)$', fontsize = 20)
ax[1,0].set_ylabel("EF", fontsize = 20)
fig.savefig("Figure, NFD effect on EF.pdf")

# imporved prediction of EF via ND and FD
fig, ax = plt.subplots(3,3, sharex = True, sharey = True, figsize = (9,9))
x = np.linspace(0,1,1000)
for i in range(len(NO_all)):
    axc = ax.flatten()[i]
    axc.scatter(np.sum((1+NO_all[i]*(FD_all[i]-1))/(1-NO_all[i]**2), axis = -1), 
                np.sum(equi_all[i], axis = -1), s = 5, linewidth = 0)
    axc.set_title(i+2)
ax[2,0].set_xlabel(r'$\overline{\mathcal{N}}+\overline{\mathcal{F}}-\overline{\mathcal{NF}}$', fontsize = 20)
ax[2,2].set_xlabel(r'$=n\left(1-NO(1-F)+cov(NO,F)\right)$', fontsize = 20)
ax[1,0].set_ylabel("EF", fontsize = 20)
fig.savefig("Figure, NFD effect on EF, improved.pdf")

#what is the dimensionality of ND and FD in lV?

fig = plt.figure()
plt.scatter(*np.log(1-FD_all[1][:,:2].T), s = 9, c = np.log(1-FD_all[1][:,2]))
#"""