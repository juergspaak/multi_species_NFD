import numpy as np
import matplotlib.pyplot as plt
import warnings

import LV_multi_functions as lmf
from interaction_estimation import resample_short


# hos links scale with species richness
b = 1.387
n_specs = np.arange(2, 51)
connectance = (n_specs/2)**b/(n_specs-1)**2

n_coms = 1000

ND_all, FD_all = np.full((2,len(n_specs),n_coms, n_specs[-1]),
                         np.nan)
for i, n in enumerate(n_specs):
    print(i)
    # create the interaction matrix
    A = resample_short(n*n*n_coms).reshape(n_coms, n,n)
    # set diagonal to 1
    A[:,np.arange(n), np.arange(n)] = 1
    
    # connectance matrix
    conn = np.random.binomial(1, (n/2)**b/(n-1)**2, A.shape)
    # links must be symetric
    ind_u = np.triu_indices(n, 1) # indices of a_ij, i<j
    conn[:, ind_u[1], ind_u[0]] = conn[:, ind_u[0], ind_u[1]]
    conn[:, np.arange(n), np.arange(n)] = 1 # sp interact with themselves
    
    A = A*conn
    
    NFD_comp, sub_equi = lmf.find_NFD_computables(A)
    A_comp = A[NFD_comp]
    sub_equi = sub_equi[NFD_comp]
    with warnings.catch_warnings(record = True):
        ND, FD, c, NO_ij, FD_ij, r_i = lmf.NFD_LV_multispecies(A_comp,sub_equi)
    ND_all[i, :len(ND), :n] = ND.copy()
    FD_all[i, :len(FD), :n] = FD.copy()
    
fig, ax = plt.subplots(2,1, sharex = True)

ax[0].plot(n_specs, np.nanmean(ND_all, axis = (1,2)), 'r', linewidth = 4)
ax[0].plot(n_specs, np.nanpercentile(ND_all, 50, axis = (1,2)), 'b')
ax[0].plot(n_specs, np.nanpercentile(ND_all, 25, axis = (1,2)), 'b--')
ax[0].plot(n_specs, np.nanpercentile(ND_all, 75, axis = (1,2)), 'b--')
ax[0].plot(n_specs, np.nanpercentile(ND_all, 5, axis = (1,2)), 'b:')
ax[0].plot(n_specs, np.nanpercentile(ND_all, 95, axis = (1,2)), 'b:')

ax[1].plot(n_specs, np.nanmean(FD_all, axis = (1,2)), 'r', linewidth = 2)
ax[1].plot(n_specs, np.nanpercentile(FD_all, 50, axis = (1,2)), 'b')
ax[1].plot(n_specs, np.nanpercentile(FD_all, 25, axis = (1,2)), 'b--')
ax[1].plot(n_specs, np.nanpercentile(FD_all, 75, axis = (1,2)), 'b--')
ax[1].plot(n_specs, np.nanpercentile(FD_all, 5, axis = (1,2)), 'b:')
ax[1].plot(n_specs, np.nanpercentile(FD_all, 95, axis = (1,2)), 'b:')
ax[1].invert_yaxis()

ax[1].set_xlim(n_specs[[0,-1]])
ax[1].set_xticks([2,10,20,30,40,50])

fs_label = 18
ax[1].set_xlabel("Species richness", fontsize = fs_label)
ax[0].set_ylabel(r"$\mathcal{N}$", fontsize = fs_label)
ax[1].set_ylabel(r"$\mathcal{F}$", fontsize = fs_label)

ax[0].set_title("A")
ax[1].set_title("B")

fig.savefig("Figure_S9.pdf")