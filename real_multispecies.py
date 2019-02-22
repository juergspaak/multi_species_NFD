"""
"""

import numpy as np
import matplotlib.pyplot as plt

from LV_real_multispec_com import LV_pars, ND_LV, FD_LV, LV_multi_spec, max_spec

fig = plt.figure(figsize = (10,10))


ax_FD_LV = fig.add_subplot(2,2,3)
ax_ND_LV = fig.add_subplot(2,2,1)

ax_ND_LV.boxplot(ND_LV[2:7], positions = range(2,7))
ax_FD_LV.boxplot(FD_LV[2:7], positions = range(2,7))
ax_ND_LV.set_ylim([-2,2.2])
ax_FD_LV.set_ylim([-20,1])

ax_coex_LV = fig.add_subplot(1,2,2)

for i in range(2,7):
    ax_coex_LV.scatter(ND_LV[i], -np.array(FD_LV[i]), s = (i+2)**2)      

y_lim = ax_coex_LV.get_ylim()
ND_bound = np.linspace(-2,2,101)
ax_coex_LV.plot(ND_bound, ND_bound/(1-ND_bound), "black")
ax_coex_LV.set_ylim([-1,10])
ax_coex_LV.set_xlim([-2,2.5])

# add layout
ax_ND_LV.set_title("Lotka Volterra")

ax_FD_LV.set_xlabel("species richness")
ax_FD_LV.set_ylabel("FD")
ax_ND_LV.set_ylabel("ND")

ax_coex_LV.set_ylabel("-FD")
ax_coex_LV.set_xlabel("ND")

fig.tight_layout()

fig.savefig("real_multispecies.pdf")

# compute number of ND we have compute
ND_comp_LV = [len(ND) for ND in ND_LV]

stab_LV = [sum(np.array(LV_pars["stable"][i]) * LV_pars["feasible"][i]) 
        for i in range(len(LV_pars["stable"]))]

tot_com_LV = [len(ND) for ND in LV_pars["feasible"]]
plt.show()
# actual measured communities
print([sum(LV_multi_spec.n_spec == i) for i in range(max_spec+1)])
# communities for which all parameters exist and are nonzero
print(tot_com_LV)
# communities which can coexist
print(stab_LV)
# communities for which we can compute NFD parameters
print(ND_comp_LV)

""" current number of communities
[0, 0, 0, 8, 10, 1, 5, 4, 2, 3]
[0, 0, 490, 629, 676, 525, 293, 113, 27, 3]
[0, 0, 376, 307, 172, 69, 13, 0, 0, 0]
[0, 0, 490, 407, 169, 60, 9, 0, 0, 0]
"""

fig, ax = plt.subplots(2,2, figsize = (9,9), sharey = "row", sharex = "col")
ax[0,0].boxplot(LV_pars["interaction_geom"][2:])
ax[1,0].boxplot(LV_pars["interaction_artm"][2:])



ax[0,1].boxplot([LV_pars["interaction_geom"][i][LV_pars["NFD_comp"][i]] for i
                 in range(2,7)])
ax[1,1].boxplot([LV_pars["interaction_artm"][i][LV_pars["NFD_comp"][i]] for i
                 in range(2,7)])
ax[1,0].set_ylim(-5,5)
ax[0,0].set_ylim(0,2)

ax[1,1].set_xlabel("Species richness")
ax[1,0].set_xlabel("Species richness")
ax[0,0].set_ylabel("Geometric mean(A)")
ax[1,0].set_ylabel("Arithemtic mean(A)")

ax[0,0].set_title("All communities")
ax[0,1].set_title("NFD_computed communities")

fig.savefig("real_communities_interaction_strenght.pdf")