"""
@author: J.W. Spaak
compute NFD values for randomly generated matrices and plot NFD distributions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import rainbow

import LV_multi_functions as lmf

def diag_fill(A, values):
    n = A.shape[-1]
    A[..., np.diag_indices(n)[0], np.diag_indices(n)[1]] = values
    return

n_spe_max = 6 # maximal number of species
n_com_prime = 10000 # number of communities at the beginning
n_coms = np.zeros(n_spe_max+1, dtype = int)
NO_all, FD_all  = np.full((2, n_spe_max+1, n_com_prime, n_spe_max), np.nan)
A_all, c_all, NO_ij_all, FD_ij_all, sub_equi_all = np.full((5, n_spe_max + 1,
                n_com_prime, n_spe_max, n_spe_max), np.nan)

max_alpha = 0.26
min_alpha = 0.08
mu = 1
n_specs = np.arange(2,n_spe_max + 1)

# number of species ranging from 2 to 7
for n in n_specs:
    # create random interaction matrices
    #A_prime = np.exp(np.random.uniform(np.log(min_alpha),np.log(max_alpha)
    #                ,size = (n_com_prime,n,n)))
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
    sub_equi_all[n, :n_coms[n], :n, :n] = sub_equi


ND_all = 1-NO_all

ND_box = [ND_all[i, :n_coms[i], :i].flatten() for i in n_specs]
FD_box = [FD_all[i, :n_coms[i], :i].flatten() for i in n_specs]

###############################################################################
# plot the results
np.seterr(divide='ignore') # division by 0 is handled correctly
fs = 14

# NO and FD versus species richness    
fig = plt.figure(figsize = (12,12))
ax_ND = fig.add_subplot(2,2,1)
ax_ND.boxplot(ND_box, positions = n_specs, showfliers = False,
              medianprops = dict(color = "black"))

ax_ND.set_ylabel(r"$\mathcal{N}$")
ax_ND.set_title("A")

A_geo_mean = A_all.copy()
diag_fill(A_geo_mean, np.nan)
alpha_geom = lmf.NFD_average(A_geo_mean[2:])

ax_ND.set_xlim(1.5, n_spe_max + 0.5)

ax_ND.plot(n_specs, 1 - alpha_geom, 'r^', markersize = 10,
              label = "prediction with constant matrix", alpha = 0.5)
ax_ND.plot(n_specs, 1 - alpha_geom, 'sb', markersize = 10,
              label = "no indirect effects", alpha = 0.5)
ax_ND.legend()

ax_FD = fig.add_subplot(2,2,3, sharex = ax_ND)
ax_FD.boxplot(FD_box, positions = n_specs, showfliers = False,
              medianprops = dict(color = "black"))
ax_FD.set_xlabel("species richness")
ax_FD.set_ylabel(r"$-\mathcal{F}$", fontsize = fs)
ax_FD.set_title("B")

ax_FD.plot(n_specs, 1-(n_specs-1)/(1+alpha_geom*(n_specs-2)), "r^",
           markersize = 10, alpha = 0.5)
ax_FD.plot(n_specs, 2 - n_specs, 'sb', markersize = 10,
              label = "no indirect effects", alpha = 0.5)

ax_FD.invert_yaxis()
ax_coex = fig.add_subplot(1,2,2)
x = np.linspace(0,1,1000)

ax_coex_LV = fig.add_subplot(1,2,2)
color = rainbow(np.linspace(0,1,len(n_specs)))
for i in n_specs:
    ax_coex.scatter(ND_all[i,:,0], FD_all[i,:,0], s = 16, alpha = 0.5,
                linewidth = 0, label = "{} species".format(i),
                c = color[i-2]*np.ones((n_coms[i],1)))
    
ax_coex.plot(x,-x/(1-x), color = "black")
ax_coex.set_xlim(np.nanpercentile(ND_all, (0,100)))
ax_coex.set_ylim(np.nanpercentile(FD_all, (1,100)))
plt.gca().invert_yaxis()
ax_coex.set_ylabel(r"$-\mathcal{F}$", fontsize = fs)
ax_coex.set_xlabel(r"$\mathcal{N}$", fontsize = fs)
ax_coex.set_title("C")

ax_coex.legend(fontsize = 12)
fig.tight_layout()
fig.savefig("Figure_NFD_sim_{}_alpha_{}.pdf".format(
    min_alpha, max_alpha))

# test the results
for test in range(5):
    n = np.random.randint(2,7)
    i = np.random.randint(n_coms[n])
    A = A_all[n,i,:n,:n]
    
    pars1 = lmf.NFD_model(lambda N: 1-A.dot(N), n_spec = n)
    a_geom =lmf.geo_mean(A_geo_mean[n,i,:n,:n])
    A_const = np.ones((n,n))*a_geom
    np.fill_diagonal(A_const, 1)
    pars2 = lmf.NFD_model(lambda N: 1-A_const.dot(N), n_spec = n)
    if np.amax(np.abs((pars1["ND"]- ND_all[n,i,:n])))>1e-5:
        raise
    print(lmf.geo_mean(pars1["ND"]), np.nanmean(pars1["ND"]), pars2["ND"][0],n)