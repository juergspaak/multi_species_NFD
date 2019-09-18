
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import viridis
from scipy.stats import linregress
from scipy.optimize import curve_fit

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
    data = pd.read_csv("test2.csv")
    #data = data[data.con == "h, "]
    
data["case"] = np.sum(data[["ord1", "ord2", "ord3", "cor", "con"]], axis = 1)
data_num = data[["ord1", "ord2", "ord3", "cor", "con"]].copy()

cases = sorted(list(set(data.case)))

def FD_fit(n, q, alpha):
    return 1 - (n-1)/(1+ (n-2)*alpha) + q*n

def ND_fit(n, q, alpha):
    return 1-alpha + q*n
n_specs = np.arange(2, n_max +1)
ND_regress = []
FD_regress = []
ND_var = []
FD_var = []
for case in cases:
    data_c = data[data.case == case]
    NFDs = np.array([data_c[["ND", "FD"]][data_c.richness == i].values
           for i in range(2, n_max +1)])
    NFDs_var = np.percentile(NFDs, [25,75], axis = 1)
    NFDs_var = NFDs_var[1]-NFDs_var[0]
    # how factors affect variance and average of NFD
    n = np.arange(3,n_max +1).reshape(-1,1)
    ND_regress.append(linregress(data_c.richness, data_c.ND)[:2])
    FD_regress.append(linregress(data_c.richness, data_c.FD)[:2])
    ND_var.append(linregress(n_specs, NFDs_var[:,0])[:2])
    FD_var.append(linregress(n_specs, NFDs_var[:,1])[:2])


results = [ND_regress, FD_regress, ND_var, FD_var]
for i, dat in enumerate(results):
    dat = np.array(dat)
    # report intercept at n = 2, not n = 0
    dat[:,1] = dat[:,1] + dat[:,0] * 2
    results[i] = dat.copy()

fig, ax = plt.subplots(2,2,figsize = (7,7))
ax[0,0].set_title("$\mathcal{N}$", fontsize = 14)
ax[0,1].set_title("$\mathcal{F}$", fontsize = 14)

ax_label = fig.add_subplot(111,frameon = False)
ax_label.tick_params(labelcolor="none", top = False,
                     bottom=False, left=False, right=False)
ax_label.set_ylabel("intercept")

ax[0,0].set_ylabel("mean")
ax[1,0].set_ylabel("variation")
ax[1,0].set_xlabel("slope")
ax[1,1].set_xlabel("slope")
s = 25
alpha = 0.5
marker = ["o",1,"+"]
c = ["blue", "red", "green"]
cases_ord1 = np.array([case[:5] for case in cases])
for i,case in enumerate(set(data.ord1)):
    index = case == cases_ord1
    ax[0,0].scatter(*results[0][index].T, s = s, 
      c = c[i], alpha = alpha, marker = marker[i])
    ax[0,1].scatter(*results[1][index].T, s = s, 
      c = c[i], alpha = alpha, marker = marker[i])
    ax[1,0].scatter(*results[2][index].T, s = s, 
      c = c[i], alpha = alpha, marker = marker[i])
    ax[1,1].scatter(*results[3][index].T, s = s, 
      c = c[i], alpha = alpha, marker = marker[i])
ax[0,0].set_xlim([-0.006, 0.003])
ax[1,0].set_xlim([-0.01, 0.001])
ax[1,0].set_ylim([0,None])
fig.savefig("figure_eff_combinations.pdf")