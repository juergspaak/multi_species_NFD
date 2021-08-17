import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import viridis
from string import ascii_uppercase as ABC

plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)
plt.rc('axes', titlesize=16)
mpl.rcParams['hatch.linewidth'] = 7.0
mpl.rcParams['hatch.color'] = "r"

n_specs = np.arange(2,7)

mu = np.array([1,1,1])
mean_cons = np.array([0.2,0.3,0.8])
var_cons = np.array([0.1,0.1,0.1])

resources = np.linspace(0,1.1,1000)
res_use = (mu*np.exp(-np.abs((mean_cons-resources[:,np.newaxis])/var_cons)**2)).T
colors = viridis(np.linspace(0,1,n_specs[-1]))[::-1]       
###############################################################################

fig = plt.figure()
ax_res = plt.gca()
ax_res.set_ylim([0,1.2])
ax_res.set_xlim(resources[[0,-1]])
ax_res.set_xticks([])
ax_res.set_yticks([])
plt.fill_between(resources, res_use[0], color = colors[0])
plt.fill_between(resources, res_use[1], color = colors[1])
plt.fill_between(resources, res_use[2], color = colors[2])
plt.fill_between(resources, res_use[0],
                            hatch = "/", edgecolor = colors[0],
                                facecolor = [1,1,1,0],
                                linewidth = 0)

fig.savefig("Figure_S6.pdf")