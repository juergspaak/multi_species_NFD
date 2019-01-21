"""
"""

import numpy as np
import matplotlib.pyplot as plt

#from AP_real_multispec_com import AP_pars, ND_AP, FD_AP
#from LV_real_multispec_com import LV_pars, ND_LV, FD_LV

fig = plt.figure(figsize = (10,10))


ax_FD_LV = fig.add_subplot(4,2,3)
ax_ND_LV = fig.add_subplot(4,2,1)

ax_ND_LV.boxplot(ND_LV[2:5], positions = range(2,5))
ax_FD_LV.boxplot(FD_LV[2:5], positions = range(2,5))

ax_ND_AP = fig.add_subplot(4,2,2)
ax_FD_AP = fig.add_subplot(4,2,4)

ax_ND_AP.boxplot(ND_AP[2:5], positions = range(2,5))
ax_FD_AP.boxplot(FD_AP[2:5], positions = range(2,5))

ax_coex_LV = fig.add_subplot(2,2,3)
ax_coex_AP = fig.add_subplot(2,2,4)

for i in range(2,5):
    ax_coex_LV.scatter(ND_LV[i], -np.array(FD_LV[i]), s = (i+2)**2)
    ax_coex_AP.scatter(ND_AP[i], -np.array(FD_AP[i]), s = (i+2)**2)

y_lim = ax_coex_LV.get_ylim()
ND_bound = np.linspace(*ax_coex_LV.get_xlim(),101)
ax_coex_LV.plot(ND_bound, ND_bound/(1-ND_bound))
ax_coex_LV.set_ylim(y_lim)

y_lim = ax_coex_AP.get_ylim()
ND_bound = np.linspace(*ax_coex_AP.get_xlim(),101)
ax_coex_AP.plot(ND_bound, ND_bound/(1-ND_bound))
ax_coex_AP.set_ylim(y_lim)

# add layout
ax_ND_LV.set_title("Lotka Volterra")
ax_ND_AP.set_title("Annual plant")

ax_FD_LV.set_xlabel("species richness")
ax_FD_LV.set_ylabel("FD")
ax_ND_LV.set_ylabel("ND")

ax_coex_LV.set_ylabel("-FD")
ax_coex_LV.set_xlabel("ND")

ax_FD_AP.set_xlabel("species richness")
ax_FD_AP.set_ylabel("FD")
ax_ND_AP.set_ylabel("ND")

ax_coex_AP.set_ylabel("-FD")
ax_coex_AP.set_xlabel("ND")

fig.tight_layout()

fig.savefig("real_multispecies.pdf")