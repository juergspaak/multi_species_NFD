import numpy as np
import matplotlib.pyplot as plt

from nfd_definitions.numerical_NFD import NFD_model

def find_real_communities(A,r_prime):
    # retain only feasible and stable communities
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
    
    real = np.full(n_com, True, dtype = bool)
    real[real] = sub_feasible & sub_stable
    return (real, A_prime[real], sub_equi[sub_feasible & sub_stable])
    
def NFD_LV_multispecies(A,sub_equi, r = 1):
    # compute the two species niche overlap
    # sqrt (a_ij*a_ji/(a_ii*a_jj))
    NO_ij = np.sign(A)*np.sqrt(np.abs(
                np.einsum("nij,nji,nii,njj->nij",A,A,1/A,1/A)))
    
    # NO is a weighted average of the two species NO
    # weights = sqrt(a_ij/a_ji*a_jj), remove a_ii by division
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

def geo_mean(A,axis = None):
    return np.exp(np.nanmean(np.log(A), axis = axis))

n_spe_max = 6 # maximal number of species
n_com_prime = 10000 # number of communities at the beginning
n_coms = np.zeros(n_spe_max+1, dtype = int)
NO_all, FD_all  = np.full((2, n_spe_max+1, n_com_prime, n_spe_max), np.nan)
A_all = np.full((n_spe_max+1, n_com_prime, n_spe_max, n_spe_max), np.nan)
equi_all = []
equi_mono = []

max_alpha = 0.3
min_alpha = 0.01
mu = 1
n_specs = np.arange(2,n_spe_max + 1)

# number of species ranging from 2 to 7
for n in n_specs:
    # create random interaction matrices
    A_prime = np.exp(np.random.uniform(np.log(min_alpha),np.log(max_alpha)
                    ,size = (n_com_prime,n,n)))
    A_prime = np.random.uniform(min_alpha, max_alpha,size = (n_com_prime,n,n))
    
    # intraspecific competition is assumed to be 1
    diag_fill(A_prime,1)
    
    # intrinsic growth rate
    r_prime = np.ones((n_com_prime,n))
    
    real, A, sub_equi = find_real_communities(A_prime, r_prime)
    n_coms[n] = len(A)
    NO, FD = NFD_LV_multispecies(A,sub_equi)
    print(len(NO),n)
    NO_all[n, :n_coms[n], :n] = NO
    FD_all[n, :n_coms[n], :n] = FD
    A_all[n, :n_coms[n], :n, :n] = A


ND_all = 1-NO_all
# check result with random index
n_test = np.random.choice(n_specs)
test_ind = np.random.randint(n_coms[n_test])
def test_f(N):
    return 1 - np.dot(A_all[n_test,test_ind, :n_test, :n_test],N)
    
pars = NFD_model(test_f,int(n_test))
print(pars["NO"])
print(NO_all[n_test,test_ind, :n_test])
print(FD_all[n_test,test_ind, :n_test])
print(pars["FD"])

ND_box = [ND_all[i, :n_coms[i], :i].flatten() for i in n_specs]
FD_box = [FD_all[i, :n_coms[i], :i].flatten() for i in n_specs]

###############################################################################
# plot the results

fs = 14

# NO and FD versus species richness    
fig = plt.figure(figsize = (11,11))
ax_ND = fig.add_subplot(2,2,1)
ax_ND.boxplot(ND_box, positions = n_specs,
              showfliers = False)

ax_ND.plot(n_specs, geo_mean(ND_all[2:], axis = (1,2)), 'ro')
ax_ND.plot(n_specs, np.nanmean(ND_all[2:], axis = (1,2)), 'ro')
ax_ND.set_ylabel(r"$\mathcal{ND}$")

alpha_geom = geo_mean(A_all[2,:,0,1])
ax_ND.set_xlim(1.5, n_spe_max + 0.5)

ax_ND.axhline(np.nanmean(ND_box[0]), color = "red", label = "theory")
ax_ND.legend()

ax_FD = fig.add_subplot(2,2,3, sharex = ax_ND)
ax_FD.boxplot(FD_box, positions = n_specs, showfliers = False)
ax_FD.set_xlabel("number of species")
ax_FD.set_ylabel(r"$\mathcal{F}$", fontsize = fs)
ax_FD.plot(n_specs, 1-geo_mean(1-FD_all[2:], axis = (1,2)), 'ro')

ax_FD.plot(n_specs, 1-(n_specs-1)/(1+alpha_geom*(n_specs-2)), color = "red")


ax_coex = fig.add_subplot(1,2,2)
x = np.linspace(0,1,1000)
if True:
    im = ax_coex.scatter(ND_all[2:, :, 0], FD_all[2:, :, 0], s = 16,
        c = n_specs.reshape(-1,1)*np.ones(NO_all[2:,:,0].shape),
        linewidth = 0, alpha = 0.5)
    ax_coex.plot(x,-x/(1-x), color = "black")
    ax_coex.set_xlim(np.nanpercentile(ND_all, (1,99)))
    ax_coex.set_ylim(np.nanpercentile(FD_all, (1,99)))
    plt.gca().invert_yaxis()
    ax_coex.set_ylabel(r"$-\mathcal{F}$", fontsize = fs)
    ax_coex.set_xlabel(r"$\mathcal{N}$", fontsize = fs)

    cbar = fig.colorbar(im,ax = ax_coex)
    cbar.ax.set_ylabel("species richness")
    fig.savefig("Figure, NFD effect on RYT,{},{}.png".format(
        min_alpha, max_alpha))
    

###############################################################################
# checkup the results

# compute geometric mean interaction strength   
NO_geom = geo_mean(NO_all[2:], axis = -1)
FD_geom = 1 - geo_mean(1-FD_all[2:], axis = -1)
geom_interact = geo_mean(A_all[2:], axis = (2,3))
geom_interact = geom_interact**(n_specs/(n_specs-1)).reshape(-1,1)
ND_diff_geom = NO_geom - geom_interact

fig, ax = plt.subplots(2,1, figsize = (7,7), sharex = True)
ax[0].boxplot(ND_diff_geom.T/(1-NO_geom.T), positions = range(2,n_spe_max +1))
ax[0].set_ylabel(r"$(ND-(1-\bar{\alpha}))/ND$")
FD_predict = (1 - (n_specs-1))/(1 + geom_interact.T*(n_specs-2))
FD_diff = (FD_geom - FD_predict.T)/FD_geom
ax[1].boxplot(FD_diff.T, positions = range(2,n_spe_max +1))

ax[1].set_xlabel("species richness")
ax[1].set_ylabel("(FD-FD_predict)/FD")
fig.savefig("Expected ND_values, geom, simulated.pdf")

# compute arithmetic mean interaction strengh
NO_artm = np.nanmean(NO_all[2:], axis = -1)
artm_interact = np.nanmean(A_all[2:], axis = (2,3))
artm_interact = artm_interact*n_specs.reshape(-1,1)**2-n_specs.reshape(-1,1)
artm_interact = artm_interact/(n_specs*(n_specs-1)).reshape(-1,1)


ND_diff_artm = NO_artm - artm_interact
fig = plt.figure()
plt.boxplot(ND_diff_geom.T/(1-NO_geom.T))
plt.xlabel("species richness")
plt.ylabel(r"$(ND-(1-\bar{\alpha}))/ND$")
fig.savefig("Expected ND_values, artm, simulated.pdf")
