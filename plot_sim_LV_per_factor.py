
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.cm import rainbow

import LV_multi_functions as lmf
###############################################################################
# create representative dataset for NFD per richness
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


# get regression data from fullfactorial design
regression = pd.read_csv("regression_fullfactorial.csv")
# add linestyle and color
color = ["red", "blue", "green", "orange"]
ls = ["-", "--", ":"]
regression["color"] = ""
regression["ls"] = ""
col_indicator = "ord1"
for i,factor in enumerate(set(regression[col_indicator])):
    regression.loc[regression[col_indicator] == factor, "color"] = color[i]
    
ls_indicator = "indirect"
for i,factor in enumerate(set(regression[ls_indicator])):
    regression.loc[regression[ls_indicator] == factor, "ls"] = ls[i]
regression.ls = "-"


###############################################################################
# plotting results
fig = plt.figure(figsize = (7,7))


ax_FD = fig.add_subplot(2,2,3)
ax_ND = fig.add_subplot(2,2,1)

# plot representative data
ax_coex = fig.add_subplot(1,2,2)
color = rainbow(np.linspace(0,1,len(n_specs)))
for i in n_specs:
    ax_coex.scatter(ND_all[i,:,0], -FD_all[i,:,0], s = 16, alpha = 0.5,
                linewidth = 0, label = "{} species".format(i),
                c = color[i-2]*np.ones((n_coms[i],1)))
    
ax_coex.legend(fontsize = 10, loc = "upper right")

# plot coexistence line
ax_coex.set_xlim([0.7, 0.95])
ND = np.array(ax_coex.get_xlim())
FD_lim = ax_coex.get_ylim()
ax_coex.plot(ND, ND/(1-ND), 'k')
ax_coex.set_ylim(FD_lim)





n = np.linspace(2,6,3)
for i, row in regression.iterrows():
    ax_ND.plot(n, row.ND_intercept + n*row.ND_slope, row.color,
               linestyle = row.ls, alpha = 0.05)
    ax_FD.plot(n, -(row.FD_intercept + n*row.FD_slope), row.color,
               linestyle = row.ls, alpha = 0.05)
    
fs = 18
fs_label = fs-2
fs_axis = fs-6

# add layout
ax_ND.set_title("A", fontsize = fs)
ax_FD.set_title("B", fontsize = fs)
ax_coex.set_title("C", fontsize = fs)

ax_coex.legend()

ax_ND.tick_params(axis='both', which='major', labelsize=fs_label)
ax_FD.tick_params(axis='both', which='major', labelsize=fs_label)
ax_coex.tick_params(axis='both', which='major', labelsize=fs_label)

ax_FD.set_xlabel("species richness",fontsize = fs_label)
ax_FD.set_ylabel(r"$-\mathcal{F}=\mathcal{E}-1$",fontsize = fs_label)
ax_ND.set_ylabel(r"$\mathcal{N}=1-\rho$",fontsize = fs_label)

ax_coex.set_ylabel(r"$-\mathcal{F}=\mathcal{E}-1$",fontsize = fs_label)
ax_coex.set_xlabel(r"$\mathcal{N}=1-\rho$",fontsize = fs_label)

ax_FD.set_xticks(np.arange(2,7))
ax_ND.set_xticklabels(len(ax_FD.get_xticks())*[""])
ax_coex.set_ylim([-1,None])
fig.tight_layout()
fig.savefig("Figure_sim_LV_per_factor.pdf")