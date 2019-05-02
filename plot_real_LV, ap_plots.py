"""
"""

import numpy as np
import matplotlib.pyplot as plt

from LV_real_multispec_com import LV_pars, max_spec

n_specs = np.arange(2,7)
fig, ax = plt.subplots(3,2, figsize = (9,9), sharey = "row", sharex = "col")
ax[0,0].boxplot(LV_pars["interaction_geom"][2:],
      positions = range(2, max_spec+1))
ax[1,0].boxplot(LV_pars["interaction_artm"][2:],
      positions = range(2, max_spec+1))
ax[2,0].boxplot(LV_pars["interaction_medi"][2:],
      positions = range(2, max_spec+1))



ax[0,1].boxplot([LV_pars["interaction_geom"][i][LV_pars["NFD_comp"][i]] for i
                 in range(2,7)], positions = range(2,7))
ax[1,1].boxplot([LV_pars["interaction_artm"][i][LV_pars["NFD_comp"][i]] for i
                 in range(2,7)], positions = range(2,7))
ax[2,1].boxplot([LV_pars["interaction_medi"][i][LV_pars["NFD_comp"][i]] for i
                 in range(2,7)], positions = range(2,7))

ax[0,0].set_ylim(0,2)
ax[1,0].set_ylim(-1,1)
ax[2,0].set_ylim(-1,1)

ax[2,1].set_xlabel("Species richness")
ax[2,0].set_xlabel("Species richness")
ax[0,0].set_ylabel("Geometric mean(A)")
ax[1,0].set_ylabel("Arithemtic mean(A)")
ax[2,0].set_ylabel("Median (A)")

ax[0,0].set_title("All communities")
ax[0,1].set_title("NFD_computed communities")

fig.savefig("Figure, real_communities_interaction_strenght.png")

fig, ax = plt.subplots(3,1, figsize = (7,7), sharex = True)
interact_comp = [LV_pars["interaction_artm"][i][LV_pars["NFD_comp"][i]] for i
                 in range(2,7)]
ND_comp = LV_pars["ND"][2:7]
FD_comp = LV_pars["FD"][2:7]
ND_av = [np.nanmean(ND, axis = 1) for ND in ND_comp]

ND_diff = [(ND_av[i]-(1-interact_comp[i]))/ND_av[i]
                 for i in range(len(ND_av))]

FD_artm = [np.nanmean(FD, axis = -1) for FD in FD_comp]
FD_predict = [(1 - (n_specs[i]-1))/(1 + 
    interact_comp[i]*(n_specs[i]-2))
    for i in range(len(n_specs))]
FD_diff_rel = [(FD_artm[i]-FD_predict[i])/np.abs(FD_artm[i])
    for i in range(len(FD_artm))]
FD_diff_rel = [FD[np.isfinite(FD)] for FD in FD_diff_rel]
ax[0].boxplot(ND_diff, positions = range(2,7))
ax[0].set_ylim([-1,1])
ax[0].set_ylabel("ND_diff")
ax[0].grid()

ax[1].boxplot(FD_diff_rel)
ax[1].set_ylim(-2,2)
ax[1].set_ylabel("FD_rel diff")
ax[1].grid()

FD_diff_abs = [(FD_artm[i]-FD_predict[i])
    for i in range(len(FD_artm))]
FD_diff_abs = [FD[np.isfinite(FD)] for FD in FD_diff_abs]
ax[2].boxplot(FD_diff_abs)
ax[2].set_ylim(-2,2)
ax[2].set_ylabel("FD abs diff")
ax[2].grid()

fig.savefig("Figure, Expected ND_values, artm, real.png")