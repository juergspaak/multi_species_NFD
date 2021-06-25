"""
@author: J.W. Spaak
compute NFD values for randomly generated matrices and plot NFD distributions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import rainbow
import pandas as pd

import LV_multi_functions as lmf
from interaction_estimation import resample_wide as resample

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

mu = 1
n_specs = np.arange(2,n_spe_max + 1)

# number of species ranging from 2 to 7
for n in n_specs:
    # create random interaction matrices
    # to have enough negative interactions
    A_prime = resample(n_com_prime*n*n).reshape((n_com_prime, n,n))
    
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

###############################################################################
# plot the results
np.seterr(divide='ignore') # division by 0 is handled correctly
fs = 14

# NO and FD versus species richness    
fig = plt.figure(figsize = (7,7))

ax_FD = fig.add_subplot(2,2,3)
ax_ND = fig.add_subplot(2,2,1)

# load the regression lines
regress = pd.read_csv("regression_fullfactorial.csv")
regress = regress[regress.ord1_strength == "strong, "]
color = {"neg, ": "green", "bot, ": "blue", "pos, ": "red"}
for i,row in regress.iterrows():
    ax_ND.plot(n_specs, row.ND_intercept + row.ND_slope*n_specs,
               color = color[row.ord1], alpha = 0.05)
    ax_FD.plot(n_specs, row.FD_intercept + row.FD_slope*n_specs,
               color = color[row.ord1], alpha = 0.05)

ax_FD.invert_yaxis()
ax_coex = fig.add_subplot(1,2,2)
x = np.linspace(0,1,1000)

ax_coex_LV = fig.add_subplot(1,2,2)
color = rainbow(np.linspace(0,1,len(n_specs)))
for i in n_specs:
    ax_coex.scatter(ND_all[i,:,0], FD_all[i,:,0], s = 16, alpha = 0.5,
                linewidth = 0, label = "{} species".format(i),
                c = color[i-2]*np.ones((n_com_prime,1)))  
 
fs = 18
fs_label = fs-2
fs_axis = fs-6    
 
ax_FD.set_ylabel(r"$\mathcal{F}$", fontsize = fs_label)
ax_FD.set_xlabel("species richness", fontsize = fs)
ax_FD.set_xticks(n_specs)
ax_ND.set_xticks(n_specs)
ax_ND.set_xticklabels([])
ax_ND.set_ylabel(r"$\mathcal{N}$", fontsize = fs_label)
ax_ND.set_title("A", fontsize = fs)
ax_FD.set_title("B", fontsize = fs)
ax_ND.set_ylim([0.7, 1.5])
ax_ND.set_yticks([0.8, 1.0, 1.2, 1.4])
ax_FD.set_yticks([0,-5,-10])
ax_FD.set_ylim([1, -10])

# create legend for FD and ND
ax_ND.plot(*np.full((2,2), np.nan), 'g', label = "pos.")
ax_ND.plot(*np.full((2,2), np.nan), 'b', label = "both")
ax_ND.plot(*np.full((2,2), np.nan), 'r', label = "neg.")
ax_ND.legend(fontsize = fs_axis-2)

ax_FD.plot(*np.full((2,2), np.nan), 'g', label = "pos.")
ax_FD.plot(*np.full((2,2), np.nan), 'b', label = "both")
ax_FD.plot(*np.full((2,2), np.nan), 'r', label = "neg.")
ax_FD.legend(fontsize = fs_axis-2)


# axis layout coexistence plot
ax_coex.plot(x,-x/(1-x), color = "black")
ax_coex.set_xlim([0,1])
ax_coex.set_xticks([0,1,2])
ax_coex.set_ylim([-20,1])
ax_coex.set_yticks([-20,-15,-10,-5,0])
ax_coex.invert_yaxis()
ax_coex.set_ylabel(r"$\mathcal{F}$", fontsize = fs_label)
ax_coex.set_xlabel(r"$\mathcal{N}$", fontsize = fs_label)
ax_coex.set_title("C")

ax_ND.tick_params(axis='both', which='major', labelsize=fs_label)
ax_FD.tick_params(axis='both', which='major', labelsize=fs_label)
ax_coex.tick_params(axis='both', which='major', labelsize=fs_label)

ax_coex.legend(fontsize = fs_axis-2)
fig.tight_layout()
fig.savefig("Figure_NFD_sim_strong.pdf")
print(np.round(np.nanmean(ND_all, axis = (1,2)),3))
