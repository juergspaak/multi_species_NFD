"""
@author: J.W.Spaak
Create the plots used in the manuscript
"""

import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# Example of Mac Arthur resource model with NO

r = np.linspace(0,10,1001)
peaks = np.array([2.3,3.7,6.7]) # peaks of the utilization functions
g = np.exp(-(r-peaks[:,np.newaxis])**2)

fig, ax = plt.subplots(1,2, figsize = (8,3), sharex = False, sharey = True)
ax[0].plot(r,g[0], "red", r,g[1], "blue", linewidth = 2)
#ax[0].plot(r, 0.5*g[0], ":r")
ax[0].fill(r,np.amin(g[:2],axis = 0), color = "purple",alpha = 0.5)
ax[0].axis([0,6,0,1.1])
ax[0].text(3,0.1,"Niche\nOverlap", ha = "center")

ax[1].plot(r, g[0], "red", r,g[1], "blue", r,g[2], "green", linewidth = 2)

ax[1].fill(r,np.amin(g[:2],axis = 0), color = "purple",alpha = 0.5)
ax[1].fill(r,np.amin(g[1:],axis = 0), color = "cyan",alpha = 0.5)

ax[0].set_xticks([])
ax[1].set_xticks([])
ax[0].set_yticks([])
ax[0].set_xlabel("Resource")
ax[1].set_xlabel("Resource")
ax[0].set_ylabel("Utilization")
ax[0].set_title("A")
ax[1].set_title("B")


fig.savefig("Figure, Utilization functions.pdf")