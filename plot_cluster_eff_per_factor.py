import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import viridis
from scipy.stats import linregress

# determine string and parameter settings for run   
ord1 = ["neg, ", "bot, ", "pos, "] # 1. order interaction
ord2 = ["neg, ", "bot, ", "pos, ", "abs, "] # second order interaction
ord3 = ["pre, ", "abs, "] # presence of third order interaction
cor = ["pos, ", "nul, ", "neg, "]


n_max = 6
factors = dict(ord1 = ord1, ord2 = ord2, ord3 = ord3, cor = cor)
try:
    data
    
except NameError:
    data = pd.read_csv("test.csv")
    #data = data[data.con == "h, "]
    
data["case"] = np.sum(data[["ord1", "ord2", "ord3", "cor", "con"]], axis = 1)
data_num = data[["ord1", "ord2", "ord3", "cor", "con"]].copy()

cases = sorted(list(set(data.case)))
n_specs = np.arange(2, n_max + 1)
fig, ax = plt.subplots(4,2, figsize = (7,12), sharex = True)

ax[0,0].set_title("$\mathcal{N}$", fontsize = 14)
ax[0,1].set_title("$\mathcal{F}$", fontsize = 14)

ax[0,0].set_ylabel("mean")
ax[1,0].set_ylabel("slope")
ax[2,0].set_ylabel("variation")
ax[3,0].set_ylabel("slope of var")

markers = {"neg, ": 1, "pos, ": "+", "bot, ": "o", "abs, ": "x", "pre, ": "o",
           "nul, ": "o", "h, ": "+", "m, ": "o", "l, ": 1}
colors = {"neg, ": "r", "pos, ": "g", "bot, ": "b", "abs, ": "k", "pre, ": "b",
           "nul, ": "b", "h, ": "g", "m, ": "b", "l, ": "r"}

for i,factor in enumerate(["ord1", "ord2", "ord3", "cor", "con"]):
    for case in list(set(data[factor])):
        print(case)
        marker = markers[case]
        color = colors[case]
        data_c = data[data[factor] == case]
        
        NFDs = np.array([data_c[["ND", "FD"]][data_c.richness == i].values
           for i in range(2, n_max +1)])
        NFDs_var = np.percentile(NFDs, [25,75], axis = 1)
        NFDs_var = (NFDs_var[1]-NFDs_var[0])
        
        for j, par in enumerate(["ND", "FD"]):
            data_save = data_c[np.isfinite(data_c[par])]
            regress = list(linregress(data_save.richness, data_save[par])[:2])
            regress[1] = regress[1] + regress[0]*2 # intercept at n = 2
            var = list(linregress(n_specs, NFDs_var[:,j])[:2])
            var[1] += var[1] + var[0]*2
            print(factor, case, j, regress)
        
            # plot the results
            ax[0,j].plot(i, regress[1], marker = marker, color = color,
              alpha = 0.8)
            ax[1,j].plot(i, regress[0], marker = marker, color = color,
              alpha = 0.8)
            ax[2,j].plot(i, var[1], marker = marker, color = color,
              alpha = 0.8)
            ax[3,j].plot(i, var[0], marker = marker, color = color,
              alpha = 0.8)

ax[-1,0].set_xticklabels(["","1st\nord", "2nd\nord", "3rd\nord", "cor", "con"])
fig.savefig("figure_cluster_eff_per_factor.pdf")