"""
@author: J.W.Spaak

Plot a figure to illustrate why ND is a weighted average, while FD is a 
weighted sum in the multispecies Lotka-Volterra case"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps

import matplotlib
matplotlib.rcParams.update({'font.size': 18})

res_line, dres = np.linspace(0,10,101, retstep = True)
def res_use(r_min, r_max, total, res_line = res_line, dres = dres):
    util = (res_line-r_min)*(r_max-res_line)
    util[(res_line<r_min) | (res_line>r_max)] = 0
    util = util/simps(util, dx = dres)* total
    return util

def plot_res_use(spec, color = "red", ax = plt):
    ax.fill(res_line, res_use(*spec), color = color, alpha = 0.5)

fig, ax = plt.subplots(2,2,figsize = (9,9), sharex = True, sharey = "row")

res_spec = [[1,7,5], [3,9,3], [0.5,6.5,1.5]]
col = ["red", "blue", "lime"]

# two species, fitness differences
plot_res_use(res_spec[0], ax = ax[0,0], color = col[0])
plot_res_use(res_spec[1], ax = ax[0,0], color = col[1])

# two species, niche differences
plot_res_use([*res_spec[0][:2], 1], ax = ax[1,0], color = col[0])
plot_res_use([*res_spec[1][:2], 1], ax = ax[1,0], color = col[1])

# three species, fitness differences
plot_res_use(res_spec[0], ax = ax[0,1], color = col[0])
plot_res_use(res_spec[1], ax = ax[0,1], color = col[1])
plot_res_use(res_spec[2], ax = ax[0,1], color = col[2])
util_comb = res_use(*res_spec[1]) + res_use(*res_spec[2])

# three species, niche differences
# compute scaling factor
scal = res_spec[1][-1]/(res_spec[1][-1]+res_spec[2][-1])
plot_res_use([*res_spec[0][:2], 1], ax = ax[1,1], color = col[0])
plot_res_use([*res_spec[1][:2], scal], ax = ax[1,1], color = col[1])
plot_res_use([*res_spec[2][:2], 1-scal], ax = ax[1,1], color = col[2])
# combined resource use
util_comb = res_use(*res_spec[1][:2], scal) + res_use(*res_spec[2][:2], 1-scal)

# change axis layout
ax[0,0].set_xlim([0,10])
ax[0,0].set_ylim([0, None])
ax[1,1].set_ylim([0, None])
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])
ax[1,1].set_yticks([])

# add labels
ax[0,0].set_title("A")
ax[0,1].set_title("B")
ax[1,0].set_title("C")
ax[1,1].set_title("D")

ax[0,0].set_ylabel("utilisation")
ax[1,0].set_ylabel("rescaled utilisation")

ax[1,0].set_xlabel("resources")
ax[1,1].set_xlabel("resources")

fig.savefig("resources_use.pdf")
