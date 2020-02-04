"""
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import rainbow
from scipy.special import comb

from LV_real_multispec_com import LV_pars, LV_multi_spec, max_spec

n_specs = np.arange(2,7) # species richness with NFD values

fig = plt.figure(figsize = (10,10))

ND_LV = LV_pars["ND"]
FD_LV = LV_pars["FD"]

ax_FD_LV = fig.add_subplot(2,2,3)
ax_ND_LV = fig.add_subplot(2,2,1)
pos = list(range(2,7))
# plot box plots of NFD distributions
ax_ND_LV.boxplot(ND_LV[2:7], positions = pos,
                 medianprops = dict(color = "black"))
ax_FD_LV.boxplot(FD_LV[2:7], positions = pos,
                 medianprops = dict(color = "black"))

ax_ND_LV.set_ylim([-1,2.5])
ax_FD_LV.set_ylim([-15,1])

# add averages based on average interaction strength
alpha_mean = [np.median(LV_pars["interaction_medi"][i][LV_pars["NFD_comp"][i]])
        for i in n_specs]
alpha_mean = np.array(alpha_mean)
ax_ND_LV.plot(n_specs, 1-alpha_mean, 'r^', markersize = 10,
              label = "Prediction with \nconst. matrix", alpha = 0.5)

ax_FD_LV.plot(n_specs, 1- (n_specs-1)/(1-(n_specs-2)*alpha_mean), 'r^',
              markersize = 10, alpha = 0.5)

# add communities without indirect effects
ax_FD_LV.plot(pos, [np.median(FD[np.isfinite(FD)]) for FD in 
                    LV_pars["FD_no_indir"][2:7]], 'sb',
                    label = "no indirect effects", alpha = 0.5)
ax_ND_LV.plot(pos, [np.median(ND[np.isfinite(ND)]) for ND in 
                    LV_pars["ND_no_indir"][2:7]], 'sb',
                    label = "no indirect effects", alpha = 0.5)
ax_FD_LV.invert_yaxis()
ax_ND_LV.legend()

ax_coex_LV = fig.add_subplot(1,2,2)
color = rainbow(np.linspace(0,1,len(n_specs)))
for i in n_specs:
    ax_coex_LV.scatter(ND_LV[i], -np.array(FD_LV[i]), s = (i+2)**2, alpha = 0.5
                       ,label = "{} species".format(i), c = color[i-2])
ax_coex_LV.legend()

y_lim = ax_coex_LV.get_ylim()
ND_bound = np.linspace(-2,2,101)
ax_coex_LV.plot(ND_bound, ND_bound/(1-ND_bound), "black")
ax_coex_LV.axhline(0, color = "grey", linestyle = "--", )
ax_coex_LV.axvline(0, color = "grey", linestyle = "--", )
ax_coex_LV.set_ylim([-1,15])
ax_coex_LV.set_xlim([-1,2.5])

# add layout
ax_ND_LV.set_title("A")
ax_FD_LV.set_title("B")
ax_coex_LV.set_title("C")

ax_FD_LV.set_xlabel("species richness")
ax_FD_LV.set_ylabel(r"$-\mathcal{F}$")
ax_ND_LV.set_ylabel(r"$\mathcal{N}$")

ax_coex_LV.set_ylabel(r"$-\mathcal{F}$")
ax_coex_LV.set_xlabel(r"$\mathcal{N}$")

# add ticks
ND_ticks, FD_ticks = [-1,0,1,2], np.array([-15,-10,-5,0,1])
ax_ND_LV.set_yticks(ND_ticks)
ax_FD_LV.set_yticks(FD_ticks)
ax_coex_LV.set_xticks(ND_ticks)
ax_coex_LV.set_yticks(-FD_ticks)

fig.tight_layout()

fig.savefig("Figure_NFD_in_LV_real.pdf")

plt.show()


###############################################################################
# simple summary for number of communities
com_sum = pd.DataFrame()
# distinct communities from papers
dist_com = [sum(LV_multi_spec.n_spec == i) for i in range(max_spec+1)]
com_sum["dist_com"] = dist_com
# maximal number of communities
com_sum["max_com"] = [int(sum(dist_com * comb(np.arange(0,max_spec +1),i)))
        for i in range(0,max_spec+1)]
# communities for which all parameters exist and are nonzero
com_sum["full_com"] = [len(mat) for mat in LV_pars["matrix"]]
# communities for which we can compute NFD parameters
[sum(comp) for comp in LV_pars["NFD_comp"]]
com_sum["NFD_comp"] = [len(ND) for ND in LV_pars["ND"]]
# communities with stable equilibrium
com_sum["coex"] = [sum(coex) for coex in LV_pars["real_coex"]]
com_sum["no_coex"] = com_sum["full_com"]-com_sum["coex"]



# number of communities, for which invasion is not possible, or does not
# predict coexistnece, but can coexist
coex_real = LV_pars["real_coex"]
NFD_comp = LV_pars["NFD_comp"]
coex_invasion = LV_pars["coex_invasion"]


coex_no_inv = [coex_real[i] & (~NFD_comp[i]) for i in n_specs]
inv_wrong = [coex_real[i][NFD_comp[i]]  != coex_invasion[i] for i in n_specs]
com_sum["no_inv"] = 0
com_sum["no_inv"].iloc[n_specs] = [sum(c) for c in coex_no_inv]
com_sum["inv_wrong"] = 0
com_sum["inv_wrong"].iloc[n_specs] = [sum(c) for c in inv_wrong]
com_sum["NFD_coex"] = com_sum["coex"]-com_sum["no_inv"]
com_sum["NFD_no_coex"] = com_sum["NFD_comp"] -com_sum["NFD_coex"]
com_sum = com_sum.T

com_sum["Total"] = np.sum(com_sum.values, axis = 1)
print(com_sum)
com_sum.index = ["Original matrices", "Subcommunities",
                 "Complete\n int. matrix", "NFD computed", "coexistence",
                 "comp. exclusion", "no invasion analysis", "invasion wrong",
                 "NFD coexistence", "NFD comp. excl"]
del(com_sum[0])
del(com_sum[1])
com_sum.to_csv("literature_data_overview.csv", index = True)


# communities in which invasion analysis is wrong
inv_wrong_com = [LV_pars["matrix"][i][LV_pars["NFD_comp"][i]][inv_wrong[i-2]]
                for i in n_specs]
"""
              0    1    2    3    4    5    6    7   8  9  total
dist_com      0    0    0    8   12    1    5    4   2  3     35
max_com      35  178  429  670  750  605  341  128  29  3   3168
full_com      0    0  423  637  678  525  293    0   0  0   2556
NFD_comp      0    0  423  441  170   58    9    0   0  0   1101
coex          0    0  345  319  175   69   13    0   0  0    921
no_coex       0    0   78  318  503  456  280    0   0  0   1635
no_inv        0    0    0   11   26   14    4    0   0  0     55
inv_wrong     0    0    0    3    0    0    0    0   0  0      3
NFD_coex      0    0  345  308  149   55    9    0   0  0    866
NFD_no_coex   0    0   78  133   21    3    0    0   0  0    235"""
