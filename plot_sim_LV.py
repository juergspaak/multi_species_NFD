"""
@author: J.W. Spaak
compute NFD values for randomly generated matrices and plot NFD distributions
"""

import numpy as np
import matplotlib.pyplot as plt

import LV_multi_functions as lmf

def diag_fill(A, values):
    n = A.shape[-1]
    A[:, np.diag_indices(n)[0], np.diag_indices(n)[1]] = values
    return

n_spe_max = 6 # maximal number of species
n_com_prime = 10000 # number of communities at the beginning
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
    ND, FD, c, NO_ij, FD_ij = lmf.NFD_LV_multispecies(A,sub_equi)
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
# plot the results
np.seterr(divide='ignore') # division by 0 is handled correctly
fs = 14

# NO and FD versus species richness    
fig = plt.figure(figsize = (11,11))
ax_ND = fig.add_subplot(2,2,1)
ax_ND.boxplot(ND_box, positions = n_specs,
              showfliers = False)

ax_ND.plot(n_specs, lmf.geo_mean(ND_all[2:], axis = (1,2)), 'ro')
ax_ND.plot(n_specs, np.nanmean(ND_all[2:], axis = (1,2)), 'ro')
ax_ND.set_ylabel(r"$\mathcal{ND}$")

alpha_geom = lmf.geo_mean(A_all[2,:,0,1])
ax_ND.set_xlim(1.5, n_spe_max + 0.5)

ax_ND.axhline(np.nanmean(ND_box[0]), color = "red", label = "theory")
ax_ND.legend()

ax_FD = fig.add_subplot(2,2,3, sharex = ax_ND)
ax_FD.boxplot(FD_box, positions = n_specs, showfliers = False)
ax_FD.set_xlabel("number of species")
ax_FD.set_ylabel(r"$\mathcal{F}$", fontsize = fs)
ax_FD.plot(n_specs, 1-lmf.geo_mean(1-FD_all[2:], axis = (1,2)), 'ro')

ax_FD.plot(n_specs, 1-(n_specs-1)/(1+alpha_geom*(n_specs-2)), color = "red")


ax_coex = fig.add_subplot(1,2,2)
x = np.linspace(0,1,1000)

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
fig.savefig("Figure, NFD in LV, sim, {}_alpha_{}.png".format(
    min_alpha, max_alpha))