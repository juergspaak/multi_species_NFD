"""
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

from LV_real_multispec_com import LV_pars, LV_multi_spec, max_spec

n_specs = np.arange(2,7) # species richness with NFD values

fig = plt.figure(figsize = (10,10))

ND_LV = LV_pars["ND"]
FD_LV = LV_pars["FD"]

ax_FD_LV = fig.add_subplot(2,2,3)
ax_ND_LV = fig.add_subplot(2,2,1)

# plot box plots of NFD distributions
ax_ND_LV.boxplot(ND_LV[2:7], positions = range(2,7))
ax_FD_LV.boxplot(FD_LV[2:7], positions = range(2,7))
ax_ND_LV.set_ylim([-2,2.2])
ax_FD_LV.set_ylim([-20,1])

# add averages based on average interaction strength
alpha_mean = [np.median(LV_pars["interaction_medi"][i][LV_pars["NFD_comp"][i]])
        for i in n_specs]
alpha_mean = np.array(alpha_mean)
ax_ND_LV.plot(n_specs, 1-alpha_mean, 'r^',
              label = "Prediction with const. matrix")
ax_ND_LV.legend()
ax_FD_LV.plot(n_specs, 1- (n_specs-1)/(1-(n_specs-2)*alpha_mean), 'r^')

ax_coex_LV = fig.add_subplot(1,2,2)
for i in n_specs:
    ax_coex_LV.scatter(ND_LV[i], -np.array(FD_LV[i]), s = (i+2)**2,
                       label = "{} species".format(i))
ax_coex_LV.legend()      

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

fig.savefig("Figure, NFD in LV, real.png")

plt.show()

# simple summary for number of communities

# actual measured communities
ac_com = [sum(LV_multi_spec.n_spec == i) for i in range(max_spec+1)]
print(ac_com, "distinct communities")
# maximal number of communities
print([int(sum(ac_com * comb(np.arange(0,max_spec +1),i))) for i
       in range(0,max_spec+1)], "all communities (with resampling)")
# communities for which all parameters exist and are nonzero
print([len(comp) for comp in LV_pars["NFD_comp"]], "all parameters")
# communities for which we can compute NFD parameters
print([len(ND) for ND in LV_pars["ND"]], "computed NFD")
# communities with stable equilibrium
print([sum(coex) for coex in LV_pars["real_coex"]], "stable equilibrium")

# number of communities, for which invasion is not possible, or does not
# predict coexistnece, but can coexist
coex_real = LV_pars["real_coex"]
NFD_comp = LV_pars["NFD_comp"]
coex_invasion = LV_pars["coex_invasion"]

coex_no_inv = [coex_real[i] & (~NFD_comp[i]) for i in n_specs]
inv_wrong = [coex_real[i][NFD_comp[i]]  != coex_invasion[i] for i in n_specs] 
print([sum(c) for c in coex_no_inv], 
       "invasion analysis not possible, but do coex")
print([sum(c) for c in inv_wrong], "invasion predicts wrong")      

# communities in which invasion analysis is wrong
inv_wrong_com = [LV_pars["matrix"][i][LV_pars["NFD_comp"][i]][inv_wrong[i-2]]
                for i in n_specs]
""" current number of communities
[0, 0, 0, 8, 10, 1, 5, 4, 2, 3] # actual communities
[33, 170, 417, 662, 748, 605, 341, 128, 29, 3] # possible communities
[0, 0, 411, 629, 676, 525, 293, 0, 0, 0] #communities with finite values
[0, 0, 411, 429, 167, 58, 9, 0, 0, 0] # communities for which we comp NFD
[0, 0, 343, 340, 209, 86, 19, 0, 0, 0] # communities with stable equilibrium
[0, 0, 396, 426, 167, 58, 9, 0, 0, 0]
"""