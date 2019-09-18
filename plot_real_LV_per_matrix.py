import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import rainbow
from scipy.stats import linregress

from LV_real_multispec_com import LV_pars, matrices, max_spec


origin = {}
spec_range = np.arange(2,7)

fig = plt.figure(figsize = (10,10))

ND_LV = LV_pars["ND"]
FD_LV = LV_pars["FD"]

ax_FD_LV = fig.add_subplot(2,2,3)
ax_ND_LV = fig.add_subplot(2,2,1)
pos = list(range(2,7))

ax_ND_LV.set_ylim([-1,2.5])
ax_FD_LV.set_ylim([-1,15])

ax_coex_LV = fig.add_subplot(1,2,2)
color = rainbow(np.linspace(0,1,len(spec_range)))
for i in spec_range:
    ax_coex_LV.scatter(ND_LV[i], -np.array(FD_LV[i]), s = (i+2)**2, alpha = 0.5
                       ,label = "{} species".format(i), c = color[i-2])
ax_coex_LV.legend()

y_lim = ax_coex_LV.get_ylim()
ND_bound = np.linspace(-2,2,101)
ax_coex_LV.plot(ND_bound, ND_bound/(1-ND_bound), "black")
ax_coex_LV.axhline(0, color = "grey", linestyle = "--")
ax_coex_LV.axvline(0, color = "grey", linestyle = "--")
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
ax_FD_LV.set_yticks(-FD_ticks)
ax_coex_LV.set_xticks(ND_ticks)
ax_coex_LV.set_yticks(-FD_ticks)

ND_reg_all = []
FD_reg_all = []
s_richness_all = []

def shapes(X):
    return np.array([len(o) for o in X])

for key in matrices.keys():
    index = [np.array(LV_pars["origin"][i]) == key for i in spec_range]
    for i in spec_range:
        index[i-2][np.array(LV_pars["NFD_comp"][i])]
    index2 = [index[i-2][np.array(LV_pars["NFD_comp"][i])] for i in spec_range]
    
    ND_org = [LV_pars["ND"][i][index2[i-2]] for i in spec_range]
    FD_org = [LV_pars["FD"][i][index2[i-2]] for i in spec_range]
    
    spec_richness = shapes(ND_org)
    s_richness = []
    ND_flat = []
    FD_flat = []
    for i in spec_range:
        s_richness += spec_richness[i-2]*i*[i]
        ND_flat += list(ND_org[i-2].flatten())
        FD_flat += list(FD_org[i-2].flatten())
        
    ND_reg_all += ND_flat
    FD_reg_all += FD_flat
    FD_flat = -np.array(FD_flat)
    s_richness_all += s_richness

    if max(s_richness) != 2:
        slope,intercept,r,p,stderr = linregress(s_richness, FD_flat)    
        ax_FD_LV.plot(s_richness, FD_flat, '.', color = "grey", alpha = 0.2)
        key_range = np.array([2, max(s_richness)])
        ax_FD_LV.plot(key_range, intercept + slope*key_range, color = "grey")
        
        slope,intercept,r,p,stderr = linregress(s_richness, ND_flat)    
        ax_ND_LV.plot(s_richness, ND_flat, '.', color = "grey", alpha = 0.2)
        key_range = np.array([2, max(s_richness)])
        ax_ND_LV.plot(key_range, intercept + slope*key_range, color = "grey")

""" #add general trend
slope,intercept,r,p,stderr = linregress(s_richness_all, FD_reg_all)    
ax_FD_LV.plot(spec_range, intercept + slope*spec_range, color = "black")

slope,intercept,r,p,stderr = linregress(s_richness_all, ND_reg_all)    
ax_ND_LV.plot(spec_range, intercept + slope*spec_range, color = "black")  
"""    
fig.tight_layout()

fig.savefig("Figure_NFD_in_LV_real_per_origin.pdf")    
    
