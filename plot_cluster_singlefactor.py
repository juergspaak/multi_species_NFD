
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import viridis

# determine string and parameter settings for run   
ord1 = ["neg, ", "bot, ", "pos, "] # 1. order interaction
ord2 = ["neg, ", "bot, ", "pos, ", "abs, "] # second order interaction
ord3 = ["pre, ", "abs, "] # presence of third order interaction
cor = ["pos, ", "nul, ", "neg, "]
con = ["h, ", "m, ", "l, "] # connectance
n_max = 6
degA = np.arange(1, n_max + 1)
factors = dict(ord1 = ord1, ord2 = ord2, ord3 = ord3, con = con, cor = cor
               ,degA = degA)
try:
    data
    
except NameError:
    data = pd.read_csv("test2.csv")
    data = data[data.con == "h, "]
    data = data[data.ord2 == "abs, "]

colors = viridis(np.linspace(0,1,4))
richness = np.array(sorted(set(data.richness)))
for key in ["ord1", "ord2", "ord3", "cor"]:
    legend = []
    fig, ax = plt.subplots(2,1,figsize = (9,9), sharex = True)
    set_key = factors[key]
    colors = viridis(np.linspace(0,1,len(set_key)))
    add = np.linspace(-0.25,0.25, len(set_key))
    for i,factor in enumerate(set_key):
        props = dict(color = colors[i], alpha = 2/len(set_key))
        ax[0].boxplot([data[(data[key] == factor) & (data.richness == j)].ND
          for j in richness], showfliers = False, positions = richness + add[i],
            boxprops = props, whiskerprops = props, widths = 0.4
            , capprops = props)
        bp = ax[1].boxplot([data[(data[key] == factor)&(data.richness == j)].FD
          for j in richness], showfliers = False, positions = richness + add[i],
            boxprops = props, whiskerprops = props, widths = 0.4
            , capprops = props)
        legend.append(bp["boxes"][0])
        
        
    ax[0].set_xlim(richness[[0,-1]] + [-1,0.5])    
    ax[0].set_title(key)
    ax[0].set_ylabel(r"$\mathcal{N}$", fontsize = 16)
    ax[1].set_ylabel(r"$\mathcal{F}$", fontsize = 16)
    ax[1].set_xlabel("species richness")
    ax[1].set_xticks(richness)
    ax[1].set_xticklabels(richness)
    ax[1].legend(legend, set_key, loc = "lower left")
    fig.savefig("Figure_cluster_{}.png".format(key))
    plt.show()
