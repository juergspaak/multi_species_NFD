"""
@author: J.W. Spaak
compute NFD values for randomly generated matrices
Compare results to simplified NFD computation using constant interspecific
interaction strength
"""

import numpy as np
import matplotlib.pyplot as plt

import LV_multi_functions as lmf

def diag_fill(A, values):
    n = A.shape[-1]
    A[:, np.diag_indices(n)[0], np.diag_indices(n)[1]] = values
    return

n_spe_max = 6 # maximal number of species
n_com_prime = 1000 # number of communities at the beginning
n_coms = np.zeros(n_spe_max+1, dtype = int)
NO_all, FD_all  = np.full((2, n_spe_max+1, n_com_prime, n_spe_max), np.nan)
A_all, c_all, NO_ij_all, FD_ij_all, sub_equi_all = np.full((5, n_spe_max + 1,
                n_com_prime, n_spe_max, n_spe_max), np.nan)

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
    
    NFD_comp, sub_equi = lmf.find_NFD_computables(A_prime, r_prime)
    A = A_prime[NFD_comp]
    sub_equi = sub_equi[NFD_comp]
    n_coms[n] = len(A)
    ND, FD, c, NO_ij, FD_ij, r_i = lmf.NFD_LV_multispecies(A,sub_equi)
    print(len(ND),n)
    NO_all[n, :n_coms[n], :n] = 1 - ND
    NO_ij_all[n, :n_coms[n], :n, :n] = NO_ij
    FD_all[n, :n_coms[n], :n] = FD
    FD_ij_all[n, :n_coms[n], :n, :n] = FD_ij
    A_all[n, :n_coms[n], :n, :n] = A
    c_all[n, :n_coms[n], :n, :n] = c
    sub_equi_all[n, :n_coms[n], :n, :n] = sub_equi[NFD_comp]


ND_all = 1-NO_all

ND_box = [ND_all[i, :n_coms[i], :i].flatten() for i in n_specs]
FD_box = [FD_all[i, :n_coms[i], :i].flatten() for i in n_specs]    

###############################################################################
# Compare actual NFD values to NFD values obtained from simple predictions
# simple predictions use constant interspecific interaction strength

# compute geometric mean interaction strength   
NO_geom = lmf.geo_mean(NO_all[2:], axis = -1)
FD_geom = 1 - lmf.geo_mean(1-FD_all[2:], axis = -1)
geom_interact = lmf.geo_mean(A_all[2:], axis = (2,3))
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
fig.savefig("Expected NFD_values, geom, simulated.pdf")

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
fig.savefig("Figure, expected NFD_vales, sim.pdf")

###############################################################################
# delete diagonal entries of c, FD_ij, NO_ij and sub_equi
c_all[:,:,np.arange(n_spe_max), np.arange(n_spe_max)] = np.nan
NO_ij_all[:,:,np.arange(n_spe_max), np.arange(n_spe_max)] = np.nan
FD_ij_all[:,:,np.arange(n_spe_max), np.arange(n_spe_max)] = np.nan
sub_equi_all[:,:,np.arange(n_spe_max), np.arange(n_spe_max)] = np.nan

# check correlation between c, N_j^* and NO_ij
fig, ax = plt.subplots(5,5, figsize = (11,11))

n_spec = 6
data = [c_all[n_spec], sub_equi_all[n_spec], NO_ij_all[n_spec],
              1-FD_ij_all[n_spec], c_all[n_spec]*sub_equi_all[n_spec]]
data = [dat[np.isfinite(dat)].flatten() for dat in data]
text = ["c", r"$N_j^{-j,*}$", "NO_ij", "1-FD_ij", r"$c_i^j\cdot N_j^{-j,*}$"]
for i in range(5):
    for j in range(5):
        if i == j:
            ax[i,j].text(0.5,0.5, text[i], fontsize = 24,
              horizontalalignment = "center", verticalalignment = "center")
        elif i < j:
            ax[i,j].hist2d(np.log(data[j]), np.log(data[i]), bins = 20)
            ax[i,j].set_facecolor("lightgreen")
        else:
            ax[i,j].hist2d(data[j], data[i], bins = 20)
            
fig.savefig("correlation for wheighted average, sim.png")


