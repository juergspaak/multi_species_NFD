"""
"""

import numpy as np
import matplotlib.pyplot as plt

from LV_real_multispec_com import LV_pars, ND_LV, FD_LV

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

fig.savefig("real_multispecies.png")

# compute number of ND we have compute
ND_comp_LV = [len(ND) for ND in ND_LV]

stab_LV = [sum(np.array(LV_pars["stable"][i]) * LV_pars["feasible"][i]) 
        for i in range(len(LV_pars["stable"]))]

tot_com_LV = [len(ND) for ND in LV_pars["feasible"]]
plt.show()
print(tot_com_LV)
print(stab_LV)
print(ND_comp_LV)